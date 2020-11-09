#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from cfgan_model import Model
# state for using GPU-1
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np

# training epoch
epoch = 300

# Whether to resume training: positive numbers for resuming epochs, and negative number for evaluation only
load = 0

# hyperparameters for target image data
target_source = 'celeba'  # mnist_train / mnist_test / cifar_train / cifar_test
channel = 1 if target_source.split('_')[0] == 'mnist' else 3
target_dim = 64
target_size = 0  # 0: whole dataset
target_batch_size = 64
feature_dim = 64


ae_loss_reg = np.linspace(0.6, 1, 300)

# hyperparameters for generating nets (Note: activations are only valid for dcgan)
generator_training_param = {'input_noise_dim': feature_dim, 'input_noise_batchsize': target_batch_size,
                            'input_noise_var': 0.3, 'net_type': 'dcgan', 'lr': 2e-4, 'inner_ite_per_batch': 1,
                            'activations': [('relu', None), ('tanh', None)], 'weight_decay': 0,
                            'lr_step_size_decay': 0, 'lr_decay_gamma': 0.5}

# Hyperparameters for adversarial nets (Note: activations_a are only valid for dcgan)
adversarial_training_param = {'input_t_dim': feature_dim, 'input_t_batchsize': target_batch_size,
                              'input_t_var': 1, 'net_type': 'dcgan', 'lr': 2e-4, 'inner_ite_per_batch': 1,
                              'activations_a': [('lrelu', 0.2), ('tanh', None)], 'weight_decay': 0,
                              'lr_step_size_decay': 0, 'lr_decay_gamma': 0.5,
                              'adv_t_sigma_num': target_batch_size,
                              'activations_t': [('lrelu', 0.2), ('tanh', None)]}

# hyperparameters for CFLossFunc
loss_alpha = 1  # amplitude
loss_beta = 1  # phase
# threshold is useless when normalization is None
loss_type = {'threshold': 'origin', 'normalization': None}
loss_threshold = 1

# training
mark = 'rcf-gan'
if adversarial_training_param['adv_t_sigma_num'] > 0:
    model_label = target_source + '_t_net_' + generator_training_param['net_type'] + '_' + mark
else:
    model_label = target_source + '_t_normal_' + generator_training_param['net_type'] + '_' + mark

model = Model(model_label,
              target_source, target_dim, target_size, target_batch_size,
              adversarial_training_param, generator_training_param,
              loss_type, loss_alpha, loss_beta, loss_threshold, ae_loss_reg,
              epoch)

if load >= 0:
    model.train(load)

# parameters for testing
num_white_noise_test = 100

# testing
model.test(num_white_noise_test)
model.test_rec()
model.save_for_scores()
# model.save_for_scores_per_epoc()
model.interpolated_imgs(8)
