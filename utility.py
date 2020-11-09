import os
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, utils


def prepare_dir(model_label):
    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('model'):
        os.mkdir('model')
    if not os.path.exists('result'):
        os.mkdir('result')

    # label dir
    model_label_dir = os.path.join('model', model_label)
    if not os.path.exists(model_label_dir):
        os.mkdir(model_label_dir)
    model_path = os.path.join(model_label_dir, 'final_model')

    # trace dir
    model_trace_dir = os.path.join(model_label_dir, 'trace')
    if not os.path.exists(model_trace_dir):
        os.mkdir(model_trace_dir)
    model_trace_path = os.path.join(model_trace_dir, 'trace_Epoch_{0}_Loss_{1:.4}')

    # result dir
    result_dir = os.path.join('result', model_label)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    model_result_trace_path = os.path.join(result_dir, 'trace_Epoch_{0}.png')
    model_result_final_dir = result_dir
    middle_result_trace_path = os.path.join(result_dir, 'Mid_trace_Epoch_{0}.png')
    return model_path, model_trace_path, model_result_trace_path, \
           model_result_final_dir, middle_result_trace_path, model_label_dir


def get_input_white_noise(white_noise_dim, white_noise_var, white_noise_num):
    white_noise = torch.randn(white_noise_num, white_noise_dim) * (white_noise_var ** 0.5)
    return white_noise


def get_input_uniform_noise(uniform_noise_upb, uniform_noise_lowb, uniform_noise_dim, uniform_noise_num):
    white_noise = np.random.uniform(low=uniform_noise_lowb, high=uniform_noise_upb,
                                    size=[uniform_noise_num, uniform_noise_dim])
    return torch.from_numpy(white_noise)


def get_target(source, target_dim, target_size, target_batch_size):
    # target_size is for the number of images used for training. 0 for using all
    if source not in ['mnist_train', 'mnist_test', 'cifar_train', 'cifar_test', 'celeba',
                      'lsunb_train', 'lsunb_test', 'lsunc_train', 'lsunc_test', 'svhn_train', 'svhn_test']:
        raise SystemExit('Error: Unknown source for target: {0}'.format(source))
    source_split = source.split('_')
    if len(source_split) == 2:
        source_split[1] = True if source_split[1] == 'train' else False
    channel = 1 if source_split[0] == 'mnist' else 3
    transform = transforms.Compose([torchvision.transforms.Resize(target_dim),
                                    transforms.CenterCrop(target_dim),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Lambda(
                                        lambda samples: (samples - 0.5) * 2),
                                    torchvision.transforms.Lambda(
                                        lambda samples: samples.view(channel, target_dim, target_dim))
                                    ])
    if source_split[0] == 'mnist':
        target = torchvision.datasets.MNIST('data', transform=transform, train=source_split[1], download=True)
    elif source_split[0] == 'cifar':
        target = torchvision.datasets.CIFAR10('data', transform=transform, train=source_split[1], download=True)
    elif source_split[0] == 'svhn':
        if source_split[1]:
            target = torchvision.datasets.SVHN('data', transform=transform, split='train', download=True)
        else:
            target = torchvision.datasets.SVHN('data', transform=transform, split='test', download=True)
    elif source_split[0] == 'lsunb':
        if source_split[1]:
            target = torchvision.datasets.LSUN('./data/lsun', classes=['bedroom_train'], transform=transform)
        else:
            target = torchvision.datasets.LSUN('./data/lsun', classes=['bedroom_val'], transform=transform)
    elif source_split[0] == 'lsunc':
        if source_split[1]:
            target = torchvision.datasets.LSUN('data/lsun/', classes=['church_outdoor_train'], transform=transform)
        else:
            target = torchvision.datasets.LSUN('data/lsun/', classes=['church_outdoor_val'], transform=transform)
    else:
        target = CELEBA(transform)
    if target_size > 0:
        target = Subset(target, [i for i in range(target_size)])

    data_loader = DataLoader(target, batch_size=target_batch_size, drop_last=True, num_workers=2,
                             shuffle=True, pin_memory=True)
    return data_loader


def CELEBA(transform):
    dataset = torchvision.datasets.ImageFolder(
        root='./data/img_align_celeba',
        transform=transform)
    return dataset


def save_data(x, dim, ax, channel, save_dir, padding=0):
    x = x / 2 + 0.5
    examples = torch.clamp(x[:64, :, :, :], 0, 1)
    ax.axis('off')
    ax.imshow(np.transpose(utils.make_grid(examples, padding=padding, normalize=True), (1, 2, 0)))
    utils.save_image(examples, save_dir, padding=padding)


def record_test(target_loader, x, dim_target, save_dir, channel):
    _, (ax1, ax2) = plt.subplots(1, 2)
    save_data(next(iter(target_loader))[0], dim_target, ax1, channel,
              os.path.join(save_dir, 'original_samples.png'), padding=2)
    save_data(x, dim_target, ax2, channel, os.path.join(save_dir, 'generated_samples.png'), padding=2)
    return


def record_test_sp(target_bacth, x, dim_target, save_dir, channel):
    _, (ax1, ax2, ax3) = plt.subplots(1, 3)
    t = torch.zeros([target_bacth.shape[0] * 2, target_bacth.shape[1], target_bacth.shape[2], target_bacth.shape[3]])
    t[0::2] = target_bacth[:]
    t[1::2] = x[:]
    save_data(target_bacth, dim_target, ax1, channel, os.path.join(save_dir, 'original_samples_ae.png'))
    save_data(x, dim_target, ax2, channel, os.path.join(save_dir, 'generated_samples_ae.png'))
    save_data(t, dim_target, ax3, channel, os.path.join(save_dir, 'all_together.png'))
    return


def record_test_interp(x, save_dir):
    x = x / 2 + 0.5
    _, ax = plt.subplots(1)
    examples = torch.clamp(x, 0, 1)
    ax.axis('off')
    ax.imshow(np.transpose(utils.make_grid(examples, padding=2, normalize=True), (1, 2, 0)))
    utils.save_image(examples, os.path.join(save_dir, 'interp_images.png'), padding=0)
    return


def record_status(x, dim_target, channel, save_dir):
    _, ax = plt.subplots(1)
    save_data(x, dim_target, ax, channel, save_dir)
    return


def tensorboard_img_writer(x, writer, name):
    examples = x[:64, :, :, :]
    examples = examples / 2 + 0.5
    writer.add_image(name, utils.make_grid(examples.clamp(0, 1), padding=2, normalize=True))


class avg_record(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
