#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from cfgan_model import Model
import argparse

# you can also state GPU here
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np

def configs(parser):
    parser.add_argument('--dataset', required=True, type=str, help='Type of dataset.',
                        choices=['cifar_train', 'cifar_test', 'celeba', 'lsunb_train', 'mnist_train', 'mnist_test',
                                 'lsunb_test', 'lsunc_train', 'lsunc_test', 'svhn_train', 'svhn_test'])
    parser.add_argument('--img_size', type=int, required=True, help='Image size in training.')

    parser.add_argument('--z_dim', type=int, default=128, help='Latent dimensions.')
    parser.add_argument('--net_model', type=str, default='dcgan', help='Net structre in training.',
                        choices=['resnet', 'dcgan'])
    parser.add_argument('--t_net', type=int, default=64, help='0 to disable t_net.')
    parser.add_argument('--bs', type=int, default=64, help='Batch sizes for all.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha weight of the amplitude in CF loss.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.')
    parser.add_argument('--mark', type=str, default='rcf-gan', help='Marks on the output dir.')
    parser.add_argument('--resume_training', type=int, default=0,
                        help='Whether to resume training; '
                             'positive numbers for resuming epochs; '
                             'negative number for evaluation only.')
    parser.add_argument('--save_period', type=int, default=10,
                        help='A period to save checkpoint.')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = configs(parser)
    args = parser.parse_args()

    # hyperparameters for RCF-GAN structure
    target_source = args.dataset
    channel = 1 if target_source.split('_')[0] == 'mnist' else 3
    target_dim = args.img_size
    target_batch_size = args.bs
    feature_dim = args.z_dim
    assert args.t_net >= 0
    t_num = args.t_net
    target_size = 0  # 0 for using the whole dataset
    ae_loss_reg = np.linspace(0.6, 1, 300)

    # hyperparameters for CFLossFunc
    loss_alpha = int(args.alpha * 2)  # amplitude
    loss_beta = int((1 - args.alpha) * 2)  # phase
    # threshold is useless when normalization is None
    loss_type = {'threshold': 'origin', 'normalization': None}
    loss_threshold = 1

    # hyperparameters for generating nets (Note: activations are only valid for dcgan)
    generator_training_param = {'input_noise_dim': feature_dim, 'input_noise_batchsize': target_batch_size,
                                'net_type': args.net_model, 'lr': args.lr,
                                # There are also some less relevant hyperparameters to you can adjust
                                'input_noise_var': 0.3, 'inner_ite_per_batch': 1,
                                # activations for the decoder/generator, only valid under dcgan structure
                                'activations': [('relu', None), ('tanh', None)],
                                'weight_decay': 0, 'lr_step_size_decay': 0, 'lr_decay_gamma': 0.5}

    # hyperparameters for adversarial nets (Note: activations_a are only valid for dcgan)
    adversarial_training_param = {'input_t_dim': feature_dim, 'input_t_batchsize': target_batch_size,
                                  'net_type': args.net_model, 'lr': args.lr, 'adv_t_sigma_num': t_num,
                                  # there are also some less relevant hyperparameters you can adjust
                                  'input_t_var': 1, 'inner_ite_per_batch': 1,
                                  # activations for the encoder/critic, only valid under dcgan
                                  'activations_a': [('lrelu', 0.2), ('tanh', None)],
                                  'weight_decay': 0, 'lr_step_size_decay': 0, 'lr_decay_gamma': 0.5,
                                  # activations for the t_net, only valid when t_net = True
                                  'activations_t': [('lrelu', 0.2), ('tanh', None)]}

    if adversarial_training_param['adv_t_sigma_num'] > 0:
        model_label = target_source + '_t_net_' + generator_training_param['net_type'] + '_' + args.mark
    else:
        model_label = target_source + '_t_normal_' + generator_training_param['net_type'] + '_' + args.mark

    # training
    model = Model(model_label,
                  target_source, target_dim, target_size, target_batch_size,
                  adversarial_training_param, generator_training_param,
                  loss_type, loss_alpha, loss_beta, loss_threshold, ae_loss_reg,
                  args.epochs)

    if args.resume_training >= 0:
        model.train(args.resume_training, args.save_period)

    # testing
    model.test()
    model.test_rec()
    model.save_for_scores()
    # model.save_for_scores_per_epoc()
    model.interpolated_imgs(8)
