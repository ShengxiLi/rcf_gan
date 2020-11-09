import time
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
from torch.optim import lr_scheduler
from ResNet import MyConvo2d
from loss_function import CFLossFunc
from network_module import SampleNet, AdvNet
from utility import prepare_dir, get_input_white_noise, get_target, record_test, record_status, avg_record, \
        record_test_sp, tensorboard_img_writer, record_test_interp
from progress.bar import Bar as Bar
from metrics.fid_score import calculate_fid_given_data
from metrics.kid_score import calculate_kid_given_data
from torch.utils.tensorboard import SummaryWriter


class Model:
    """
    RCF-GAN module
    Args:
        model_label: the name of output directories
        target_source: name of training dataset
        target_dim: image size
        target_batch_size: training batch size
        adversarial_training_param: a dictionary that contains training hyperparameters for the critic
        generator_training_param: a dictionary that contains training hyperparameters for the generator
        loss_type: a specification of choosing types of CF loss
        loss_alpha: the weight for amplitude in CF loss, from 0-1
        loss_beta: the weight for phase in CF loss, from 0-1
        threshold: to cut off some small weights for normalisation, we did not use this in the current version
                    please refer to https://link.springer.com/chapter/10.1007/978-3-030-30487-4_27 for more details
        loss_reg: the regularisation on reciprocal requirement
        epoch: training epoch

    """
    def __init__(self, model_label,
                 target_source, target_dim, target_size, target_batch_size,
                 adversarial_training_param, generator_training_param,
                 loss_type, loss_alpha, loss_beta, threshold, loss_reg,
                 epoch):
        # init dir
        self.model_path, self.model_trace_path, self.result_path, \
        self.test_result_dir, self.mid_result_path, self.model_file = prepare_dir(model_label)

        # white noise -> x
        self.white_noise_dim = generator_training_param['input_noise_dim']
        self.white_noise_var = generator_training_param['input_noise_var']
        self.white_noise_batch_size = generator_training_param['input_noise_batchsize']

        # target
        if target_source.split('_')[0] == 'mnist':
            self.target_channel = 1
        else:
            self.target_channel = 3
        self.target_dim = target_dim
        self.target_source = target_source
        self.target_loader = get_target(target_source, target_dim, target_size, target_batch_size)

        self.gen_net_type = generator_training_param['net_type']
        self.adv_net_type = adversarial_training_param['net_type']
        if generator_training_param['net_type'] not in ['dcgan', 'resnet']:
            raise SystemExit('Error: The generator is only supported for dcgan, adv-dcgan and resnet structures. '
                             'Unknown source for target: {0}'.format(generator_training_param['net_type']))
        if adversarial_training_param['net_type'] not in ['dcgan', 'resnet']:
            raise SystemExit('Error: The critic is only supported for dcgan and resnet structures. '
                             'Unknown source for target: {0}'.format(generator_training_param['net_type']))

        # Generator Net
        self.sample_net = nn.DataParallel(SampleNet(self.white_noise_dim, self.target_dim, self.target_channel,
                                                    generator_training_param))

        # Critic Net
        self.adversarial_net = nn.DataParallel(AdvNet(self.target_channel, self.target_dim, self.white_noise_dim,
                                                      adversarial_training_param))
        print('Generator net:', self.sample_net)
        print('Critic net:', self.adversarial_net)

        # loss function: CFLossFun
        self.loss_fun = CFLossFunc(loss_type, loss_alpha, loss_beta, threshold)

        # optimization for generator
        self.optimizer_gen = optim.Adam(self.sample_net.parameters(), lr=generator_training_param['lr'],
                                        betas=(0.5, 0.999), weight_decay=generator_training_param['weight_decay'])
        self.lr_decay_gen = False
        if generator_training_param['lr_step_size_decay'] > 0:
            self.lr_decay_gen = True
            self.lr_scheduler_gen = lr_scheduler.StepLR(self.optimizer_gen,
                                                        step_size=generator_training_param['lr_step_size_decay'],
                                                        gamma=generator_training_param['lr_decay_gamma'])

        # optimization for adversarial
        self.optimizer_adv = optim.Adam(self.adversarial_net.parameters(), lr=adversarial_training_param['lr'],
                                        betas=(0.5, 0.999), weight_decay=adversarial_training_param['weight_decay'])
        self.lr_decay_adv = False
        if adversarial_training_param['lr_step_size_decay'] > 0:
            self.lr_decay_adv = True
            self.lr_scheduler_adv = lr_scheduler.StepLR(self.optimizer_adv,
                                                        step_size=adversarial_training_param['lr_step_size_decay'],
                                                        gamma=adversarial_training_param['lr_decay_gamma'])

        self.epoch = epoch
        self.lamda = loss_reg
        self.inner_ite_gen = generator_training_param['inner_ite_per_batch']
        self.inner_ite_adv = adversarial_training_param['inner_ite_per_batch']
        # tensorboard
        self.writer = SummaryWriter('runs/' + model_label)

    def train(self, load, sp=10):
        # running statistics avg moving
        ae_loss_latent = avg_record()
        ae_loss_z = avg_record()
        # cuda setting and load previous model
        device = 'cpu'
        if torch.cuda.is_available():
            device = torch.device('cuda')
        if load > 0:
            model_list = os.listdir(self.model_file + '/trace')
            cp_path = None
            for i in range(self.epoch):
                if model_list[i].startswith('trace_Epoch_' + str(load) + '_Loss'):
                    cp_path = self.model_file + '/trace/' + model_list[i]
                    break
            assert (cp_path is not None)
            cp = torch.load(cp_path, map_location=device)
            start_epoch = cp['epoch']
            self.sample_net.load_state_dict(cp['gen_model_state_dict'])
            self.sample_net.to(device)
            self.adversarial_net.load_state_dict(cp['adv_model_state_dict'])
            self.adversarial_net.to(device)
            self.optimizer_adv.load_state_dict(cp['adv_optimizer_state_dict'])
            self.optimizer_gen.load_state_dict(cp['gen_optimizer_state_dict'])
        else:
            start_epoch = 0
            # initilize generator
            self.sample_net.to(device)
            if self.gen_net_type == 'resnet':
                self.sample_net.apply(weights_init_resnet)
            else:
                self.sample_net.apply(weights_init_dcgan)

            # initilize adv
            self.adversarial_net.to(device)
            if self.adv_net_type == 'resnet':
                self.adversarial_net.apply(weights_init_resnet)
            else:
                self.adversarial_net.apply(weights_init_dcgan)

        # train
        start = time.time()
        # x, loss = None, None
        self.adversarial_net.train()
        self.sample_net.train()
        for i in range(start_epoch, self.epoch):
            # initislise bar progress plot, each epoch is restricted to 500 iterations
            bar = Bar('Training', max=min(len(self.target_loader), 500))
            ae_loss_latent.reset()
            ae_loss_z.reset()
            if i < 300:
                lam_d = self.lamda[i]
            else:
                lam_d = self.lamda[-1]
            # mini batch
            for i_batch, target_batch in enumerate(self.target_loader):
                target_batch = target_batch[0].detach().to(device)

                # train t_nets
                self.optimizer_adv.zero_grad()
                for i_adv in range(self.inner_ite_adv):
                    white_noise = get_input_white_noise(self.white_noise_dim, self.white_noise_var,
                                                        self.white_noise_batch_size).detach().to(device)
                    # spherical distribution
                    # white_noise = nn.functional.normalize(white_noise, p=2, dim=1)
                    # forward again
                    with torch.no_grad():
                        x = self.sample_net(white_noise)
                    x_feature = self.adversarial_net(x.detach())
                    target_feature = self.adversarial_net(target_batch)
                    t_batch = self.adversarial_net.module.net_t()
                    latent_ae_loss = self.loss_fun(t_batch, white_noise, target_feature)
                    latent_ae_loss_z = nn.functional.mse_loss(x_feature, white_noise)
                    negloss = latent_ae_loss - self.loss_fun(t_batch, white_noise, x_feature) \
                              + lam_d * latent_ae_loss_z
                    negloss = negloss / self.inner_ite_adv
                    negloss.backward()
                    with torch.no_grad():
                        ae_loss_latent.update(1e4 * latent_ae_loss)  # real embedding distance
                        ae_loss_z.update(1e4 * latent_ae_loss_z)  # reciprocal loss
                self.optimizer_adv.step()

                # training generator/sampler
                # train gan loss
                self.optimizer_gen.zero_grad()
                for i_gen in range(self.inner_ite_gen):
                    white_noise = get_input_white_noise(self.white_noise_dim, self.white_noise_var,
                                                        self.white_noise_batch_size).detach().to(device)
                    # spherical distribution
                    # white_noise = nn.functional.normalize(white_noise, p=2, dim=1)
                    with torch.no_grad():
                        t_batch = self.adversarial_net.module.net_t()
                        target_feature = self.adversarial_net(target_batch)
                        target_mimic = self.sample_net(target_feature.detach())
                    # forward
                    x = self.sample_net(white_noise)
                    x_feature = self.adversarial_net(x)
                    # compute loss amd backward
                    g_loss = self.loss_fun(t_batch.detach(), x_feature, target_feature.detach())
                    g_loss = g_loss / self.inner_ite_gen
                    g_loss.backward()
                # Update
                self.optimizer_gen.step()

                # plot progress
                bar.suffix = '({batch}/{size}) | Epoc:{ep:.1f} | Time:{total:} | ETA:{eta:} | RED:{aeloss_latent:.1f} ' \
                             '| Recip:{aeloss_z:.1f} |'.format(
                    batch=i_batch + 1,
                    size=min(len(self.target_loader), 500),
                    ep=i,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    aeloss_latent=ae_loss_latent.avg,
                    aeloss_z=ae_loss_z.avg,
                )
                bar.next()
                if i_batch > 498:
                    # break to keep that each epoch trains a same number of g ites
                    break
            bar.finish()

            # update scheduler information
            if self.lr_decay_gen:
                self.lr_scheduler_gen.step()
            if self.lr_decay_adv:
                self.lr_scheduler_adv.step()

            self.writer.add_scalar('real embedding loss', ae_loss_latent.avg, i + 1)
            self.writer.add_scalar('reciprocal loss', ae_loss_z.avg, i + 1)

            if i == 0 or (i + 1) % sp == 0:
                epoch_count = i + 1
                time_diff = time.time() - start
                print('Epoch {0}: {1:.4}, time: {2:.4f}'.format(epoch_count,
                                                                (ae_loss_latent.avg + ae_loss_z.avg) / 1e4, time_diff))
                # save middle checkpoints
                torch.save({
                    'epoch': epoch_count,
                    'adv_model_state_dict': self.adversarial_net.state_dict(),
                    'adv_optimizer_state_dict': self.optimizer_adv.state_dict(),
                    'gen_model_state_dict': self.sample_net.state_dict(),
                    'gen_optimizer_state_dict': self.optimizer_gen.state_dict(),
                }, self.model_trace_path.format(epoch_count, (ae_loss_latent.avg + ae_loss_z.avg) / 1e4))
                with torch.no_grad():
                    record_status(x.to(torch.device('cpu')).detach(), self.target_dim,
                                  self.target_channel, self.result_path.format(epoch_count))
                    tensorboard_img_writer(x.to(torch.device('cpu')).detach(), self.writer, 'generated images')
                # record reconstruction results
                with torch.no_grad():
                    record_status(target_mimic.to(torch.device('cpu')).detach(), self.target_dim,
                                  self.target_channel, self.mid_result_path.format(epoch_count))
                    tensorboard_img_writer(target_mimic.to(torch.device('cpu')).detach(),
                                           self.writer, 'reconstructed images')

        # # save model
        torch.save({
            'adv_model_state_dict': self.adversarial_net.state_dict(),
            'gen_model_state_dict': self.sample_net.state_dict(),
        }, self.model_path)

    def test(self, num_white_noise_test=100):
        # Test to evaluate random generation
        # cuda setting
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load model
        cp = torch.load(self.model_path, map_location=device)
        self.sample_net.load_state_dict(cp['gen_model_state_dict'])
        self.sample_net.to(device)
        # manually set to evaluation mode because of some pytorch bugs in evaluation mode, please refer to
        # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/43
        for child in self.sample_net.module.children():
            if self.gen_net_type == 'resnet':
                if type(child).__name__.startswith('BatchNorm'):
                    child.track_running_stats = False
            else:
                for ii in range(len(child)):
                    if type(child[ii]).__name__.startswith('BatchNorm'):
                        child[ii].track_running_stats = False

        # samples generated from model
        test_white_noise = get_input_white_noise(self.white_noise_dim, self.white_noise_var,
                                                 num_white_noise_test).to(device)
        # test_white_noise = nn.functional.normalize(test_white_noise, p=2, dim=1)
        x = self.sample_net(test_white_noise).to(torch.device('cpu')).detach()
        # plot result
        record_test(self.target_loader, x, self.target_dim, self.test_result_dir, self.target_channel)
        tensorboard_img_writer(x.to(torch.device('cpu')).detach(), self.writer, 'test generated images')

    def test_rec(self):
        # Test to evaluate reconstructions and also FID and KID scores of reconstructed images
        # cuda setting
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load model
        cp = torch.load(self.model_path, map_location=device)
        self.sample_net.load_state_dict(cp['gen_model_state_dict'])
        self.sample_net.to(device)
        # manually set to evaluation mode because of some pytorch bugs in evaluation mode, please refer to
        # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/43
        for child in self.sample_net.module.children():
            if self.gen_net_type == 'resnet':
                if type(child).__name__.startswith('BatchNorm'):
                    child.track_running_stats = False
            else:
                for ii in range(len(child)):
                    if type(child[ii]).__name__.startswith('BatchNorm'):
                        child[ii].track_running_stats = False

        self.adversarial_net.load_state_dict(cp['adv_model_state_dict'])
        self.adversarial_net.to(device)
        # manually set to evaluation mode because of some pytorch bugs in evaluation mode, please refer to
        # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/43
        for child in self.adversarial_net.module.children():
            if self.adv_net_type == 'resnet':
                if type(child).__name__.startswith('BatchNorm'):
                    child.track_running_stats = False
            else:
                for ii in range(len(child)):
                    if type(child[ii]).__name__.startswith('LayerNorm'):
                        child[ii].track_running_stats = False

        # samples generated from model
        if self.target_source.split('_')[0] == 'celeba':
            target_loader = get_target(self.target_source, self.target_dim, 0, 100)
        else:
            # use validations for out-sample validation
            target_loader = get_target(self.target_source.split('_')[0] + '_test', self.target_dim, 0, 100)

        with torch.no_grad():
            target_batch = next(iter(target_loader))[0].to(device)
            feature = self.adversarial_net(target_batch)
            # feature = nn.functional.normalize(feature, p=2, dim=1)
            x = self.sample_net(feature).to(torch.device('cpu')).detach()

            # plot result
            record_test_sp(target_batch.to(torch.device('cpu')).detach(), x, self.target_dim,
                           self.test_result_dir, self.target_channel)
            tensorboard_img_writer(x.to(torch.device('cpu')).detach(), self.writer, 'test reconstructed images')
            tensorboard_img_writer(target_batch.to(torch.device('cpu')).detach(), self.writer, 'test original images')

        # test systematically
        rec_imgs = []
        real_imgs = []
        # here use the training set as the test do not have enough samples (25000) to get accurate FID and KID
        target_loader = get_target(self.target_source, self.target_dim, 0, 100)
        for i_batch, target_batch in enumerate(target_loader):
            x = target_batch[0].to(torch.device('cpu')).detach()
            real_imgs.append(x)
            with torch.no_grad():
                feature = self.adversarial_net(target_batch[0].to(device).detach())
                # feature = nn.functional.normalize(feature, p=2, dim=1)
                x = self.sample_net(feature).to(torch.device('cpu')).detach()
            rec_imgs.append(x)
            if i_batch > 248:
                break
        real_imgs = torch.cat(real_imgs, dim=0)
        rec_imgs = torch.cat(rec_imgs, dim=0)
        rec_imgs = rec_imgs.numpy()
        real_imgs = real_imgs.numpy()
        # test for scores
        if self.target_source.split('_')[0] == 'mnist':
            model_type = 'lenet'
        else:
            model_type = 'inception'
        # for fid
        results = calculate_fid_given_data(rec_imgs, real_imgs, 100, torch.cuda.is_available(), 2048,
                                           model_type=model_type)
        for m, s in results:
            print('FID for Reconstruction: %.2f (%.3f)' % (m, s))
            self.writer.add_scalar('rec_fid_mean:', m, 1)
            self.writer.add_scalar('rec_fid_std:', s, 1)

        # for kid
        results = calculate_kid_given_data(rec_imgs, real_imgs, 100, torch.cuda.is_available(), 2048,
                                           model_type=model_type)
        for m, s in results:
            print('KID for Reconstruction: %.3f (%.3f)' % (m, s))
            self.writer.add_scalar('rec_kid_mean:', m, 1)
            self.writer.add_scalar('rec_kid_std:', s, 1)

    def save_for_scores(self, total_images=25000, batch_size=100):
        # Test to calculate KID and FID scores for the randomly generated images
        # cuda setting
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load model
        cp = torch.load(self.model_path, map_location=device)
        self.sample_net.load_state_dict(cp['gen_model_state_dict'])
        self.sample_net.to(device)
        # manually set to evaluation mode because of some pytorch bugs in evaluation mode, please refer to
        # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/43
        for child in self.sample_net.module.children():
            if self.gen_net_type == 'resnet':
                if type(child).__name__.startswith('BatchNorm'):
                    child.track_running_stats = False
            else:
                for ii in range(len(child)):
                    if type(child[ii]).__name__.startswith('BatchNorm'):
                        child[ii].track_running_stats = False

        # samples generated from model
        n_batches = total_images // batch_size
        generated_imgs = []
        for i in range(n_batches):
            with torch.no_grad():
                test_white_noise = get_input_white_noise(self.white_noise_dim, self.white_noise_var, batch_size).to(
                    device)
                # test_white_noise = nn.functional.normalize(test_white_noise, p=2, dim=1)
                x = self.sample_net(test_white_noise).to(torch.device('cpu')).detach()
            generated_imgs.append(x)
        generated_imgs = torch.cat(generated_imgs, dim=0)
        generated_imgs = generated_imgs.numpy()

        # samples for true data
        target_loader = get_target(self.target_source, self.target_dim, 0, batch_size)
        true_imgs = []
        for i_batch, target_batch in enumerate(target_loader):
            x = target_batch[0].to(torch.device('cpu')).detach()
            true_imgs.append(x)
            if i_batch > n_batches:
                break
        true_imgs = torch.cat(true_imgs, dim=0)
        true_imgs = true_imgs.numpy()

        # test for scores
        if self.target_source.split('_')[0] == 'mnist':
            model_type = 'lenet'
        else:
            model_type = 'inception'
        # for fid
        results = calculate_fid_given_data(generated_imgs, true_imgs, batch_size, torch.cuda.is_available(), 2048,
                                           model_type=model_type)
        for m, s in results:
            print('FID for Random Generation: %.2f (%.3f)' % (m, s))
            self.writer.add_scalar('fid_mean:', m, 1)
            self.writer.add_scalar('fid_std:', s, 1)

        # for kid
        results = calculate_kid_given_data(generated_imgs, true_imgs, batch_size, torch.cuda.is_available(), 2048,
                                           model_type=model_type)
        for m, s in results:
            print('KID for Random Generation: %.3f (%.3f)' % (m, s))
            self.writer.add_scalar('kid_mean:', m, 1)
            self.writer.add_scalar('kid_std:', s, 1)

    def save_for_scores_per_epoc(self, total_images=25000, batch_size=100):
        # You can also use this function to plot KID and FID scores for every checkpoint
        # cuda setting
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_list = os.listdir(self.model_file + '/trace')
        for epoch in range(10, self.epoch + 1, 10):
            # load model
            for i in range(len(model_list)):
                if model_list[i].startswith('trace_Epoch_' + str(epoch) + '_Loss'):
                    cp = torch.load(self.model_file + '/trace/' + model_list[i], map_location=device)
            self.sample_net.load_state_dict(cp['gen_model_state_dict'])
            self.sample_net.to(device)

            # manually set to evaluation mode because of some pytorch bugs in evaluation mode, please refer to
            # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/43
            for child in self.sample_net.module.children():
                if self.gen_net_type == 'resnet':
                    if type(child).__name__.startswith('BatchNorm'):
                        child.track_running_stats = False
                else:
                    for ii in range(len(child)):
                        if type(child[ii]).__name__.startswith('BatchNorm'):
                            child[ii].track_running_stats = False

            # samples generated from model
            n_batches = total_images // batch_size
            generated_imgs = []
            for i in range(n_batches):
                with torch.no_grad():
                    test_white_noise = get_input_white_noise(self.white_noise_dim, self.white_noise_var, batch_size).to(
                        device)
                    # test_white_noise = nn.functional.normalize(test_white_noise, p=2, dim=1)
                    x = self.sample_net(test_white_noise).to(torch.device('cpu')).detach()
                generated_imgs.append(x)
            generated_imgs = torch.cat(generated_imgs, dim=0)
            generated_imgs = generated_imgs.numpy()

            # samples for true data
            target_loader = get_target(self.target_source, self.target_dim, 0, batch_size)
            true_imgs = []
            for i_batch, target_batch in enumerate(target_loader):
                x = target_batch[0].to(torch.device('cpu')).detach()
                true_imgs.append(x)
                if i_batch > n_batches - 2:
                    break
            true_imgs = torch.cat(true_imgs, dim=0)
            true_imgs = true_imgs.numpy()

            # test for scores
            if self.target_source.split('_')[0] == 'mnist':
                model_type = 'lenet'
            else:
                model_type = 'inception'
            # for fid
            results = calculate_fid_given_data(generated_imgs, true_imgs, batch_size, torch.cuda.is_available(), 2048,
                                               model_type=model_type)
            for m, s in results:
                print('FID for Epoch %.2f: %.2f (%.3f)' % (epoch, m, s))
                self.writer.add_scalar('fid_mean:', m, epoch)
                self.writer.add_scalar('fid_std:', s, epoch)

            # for kid
            results = calculate_kid_given_data(generated_imgs, true_imgs, batch_size, torch.cuda.is_available(), 2048,
                                               model_type=model_type)
            for m, s in results:
                print('KID for Epoch %.2f: %.3f (%.3f)' % (epoch, m, s))
                self.writer.add_scalar('kid_mean:', m, epoch)
                self.writer.add_scalar('kid_std:', s, epoch)

    def interpolated_imgs(self, num_of_groups):
        # this function is used for interpolation between images

        # cuda setting
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load model
        cp = torch.load(self.model_path, map_location=device)
        self.sample_net.load_state_dict(cp['gen_model_state_dict'])
        self.sample_net.to(device)
        # manually set to evaluation mode because of some pytorch bugs in evaluation mode, please refer to
        # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/43
        for child in self.sample_net.module.children():
            if self.gen_net_type == 'resnet':
                if type(child).__name__.startswith('BatchNorm'):
                    child.track_running_stats = False
            else:
                for ii in range(len(child)):
                    if type(child[ii]).__name__.startswith('BatchNorm'):
                        child[ii].track_running_stats = False

        self.adversarial_net.load_state_dict(cp['adv_model_state_dict'])
        self.adversarial_net.to(device)
        # manually set to evaluation mode because of some pytorch bugs in evaluation mode, please refer to
        # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/43
        for child in self.adversarial_net.module.children():
            if self.adv_net_type == 'resnet':
                if type(child).__name__.startswith('BatchNorm'):
                    child.track_running_stats = False
            else:
                for ii in range(len(child)):
                    if type(child[ii]).__name__.startswith('LayerNorm'):
                        child[ii].track_running_stats = False

        if self.target_source.split('_')[0] == 'celeba':
            target_loader = get_target(self.target_source, self.target_dim, 0, 100)
        else:
            # use validations for out-sample validation
            target_loader = get_target(self.target_source.split('_')[0] + '_test', self.target_dim, 0, 100)

        target_batch = next(iter(target_loader))[0].to(device)
        feature = self.adversarial_net(target_batch)

        # interpolation
        interp_imgs = torch.zeros([num_of_groups * 8, target_batch.shape[1], target_batch.shape[2],
                                   target_batch.shape[3]])
        interp_feature = torch.zeros([num_of_groups * 6, feature.shape[1]]).to(device)
        for i in range(num_of_groups):
            for j in range(6):
                alpha = j / 5
                interp_feature[(i - 1) * 6 + j, :] = alpha * feature[(i - 1) * 2 + 1, :] \
                                                     + (1 - alpha) * feature[(i - 1) * 2, :]

        # interp_feature = nn.functional.normalize(interp_feature, p=2, dim=1)
        x = self.sample_net(interp_feature).to(torch.device('cpu')).detach()
        for i in range(num_of_groups):
            interp_imgs[(i - 1) * 8, :, :, :] = target_batch[(i - 1) * 2, :, :, :]
            interp_imgs[(i - 1) * 8 + 7, :, :, :] = target_batch[(i - 1) * 2 + 1, :, :, :]
            for j in range(6):
                interp_imgs[(i - 1) * 8 + j + 1, :, :, :] = x[(i - 1) * 6 + j, :, :, :]
        # plot result
        record_test_interp(interp_imgs, self.test_result_dir)
        tensorboard_img_writer(interp_imgs, self.writer, 'Interpolated images')


def weights_init_dcgan(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        if m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        # m.bias.data.fill_(0)


def weights_init_resnet(m):
    if isinstance(m, MyConvo2d):
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)
