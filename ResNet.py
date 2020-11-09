# This file is adapted from https://github.com/jalola/improved-wgan-pytorch/blob/master/models/wgan.py

from torch import nn
from torch.autograd import grad
import torch

import pdb


class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True, stride=1, bias=True, sn=False):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=self.padding, bias=bias)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, input):
        output = self.conv(input)
        return output


class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True, sn=False):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init=self.he_init, sn=sn)

    def forward(self, input):
        output = self.conv(input)
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2]
                  + output[:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4
        return output


class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True, sn=False):
        super(MeanPoolConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init=self.he_init, sn=sn)

    def forward(self, input):
        output = input
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2]
                  + output[:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4
        output = self.conv(output)
        return output


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size, input_height, output_width, output_depth) for t_t in spl]
        output = torch.stack(stacks, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, output_height,
                                                                                       output_width, output_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class UpSampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init=True, bias=True):
        super(UpSampleConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init=self.he_init, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, hw=64, sn=False):
        super(ResidualBlock, self).__init__()
        self.sn = sn
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if resample == 'down':
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        elif resample == 'up':
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        elif resample == None:
            self.bn1 = nn.BatchNorm2d(output_dim)
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        else:
            raise Exception('invalid resample value')

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size=1, he_init=False, sn=sn)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size=kernel_size, bias=False, sn=sn)
            self.conv_2 = ConvMeanPool(input_dim, output_dim, kernel_size=kernel_size, sn=sn)
        elif resample == 'up':
            self.conv_shortcut = UpSampleConv(input_dim, output_dim, kernel_size=1, he_init=False)
            self.conv_1 = UpSampleConv(input_dim, output_dim, kernel_size=kernel_size, bias=False)
            self.conv_2 = MyConvo2d(output_dim, output_dim, kernel_size=kernel_size)
        elif resample == None:
            self.conv_shortcut = MyConvo2d(input_dim, output_dim, kernel_size=1, he_init=False, sn=sn)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size=kernel_size, bias=False, sn=sn)
            self.conv_2 = MyConvo2d(input_dim, output_dim, kernel_size=kernel_size, sn=sn)
        else:
            raise Exception('invalid resample value')

    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample == None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)

        output = input
        if self.sn:
            output = self.relu1(output)
            output = self.conv_1(output)
            output = self.relu2(output)
            output = self.conv_2(output)
        else:
            output = self.bn1(output)
            output = self.relu1(output)
            output = self.conv_1(output)
            output = self.bn2(output)
            output = self.relu2(output)
            output = self.conv_2(output)

        return shortcut + output

# TO DO: write GoodGenerator64 and 128 in a unified form as in the DCGAN
class GoodGenerator64(nn.Module):
    def __init__(self, input_dim=128, dim=64):
        super(GoodGenerator64, self).__init__()

        self.dim = dim

        self.ssize = self.dim // 16
        self.ln1 = nn.Linear(input_dim, self.ssize * self.ssize * 8 * self.dim)
        self.rb1 = ResidualBlock(8 * self.dim, 8 * self.dim, 3, resample='up')
        self.rb2 = ResidualBlock(8 * self.dim, 4 * self.dim, 3, resample='up')
        self.rb3 = ResidualBlock(4 * self.dim, 2 * self.dim, 3, resample='up')
        self.rb4 = ResidualBlock(2 * self.dim, 1 * self.dim, 3, resample='up')
        self.bn = nn.BatchNorm2d(self.dim)

        self.conv1 = MyConvo2d(1 * self.dim, 3, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.ln1(input.contiguous())
        output = output.view(-1, 8 * self.dim, self.ssize, self.ssize)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)

        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.tanh(output)
        return output

# TO DO: write GoodDiscriminator64 and 128 in a unified form as in the DCGAN
class GoodDiscriminator64(nn.Module):
    def __init__(self, dim=64, output_dim=128, sn=False):
        super(GoodDiscriminator64, self).__init__()

        self.dim = dim

        self.ssize = self.dim // 16
        self.conv1 = MyConvo2d(3, self.dim, 3, he_init=False, sn=sn)
        self.rb1 = ResidualBlock(self.dim, 2 * self.dim, 3, resample='down', hw=self.dim, sn=sn)
        self.rb2 = ResidualBlock(2 * self.dim, 4 * self.dim, 3, resample='down', hw=int(self.dim / 2), sn=sn)
        self.rb3 = ResidualBlock(4 * self.dim, 8 * self.dim, 3, resample='down', hw=int(self.dim / 4), sn=sn)
        self.rb4 = ResidualBlock(8 * self.dim, 8 * self.dim, 3, resample='down', hw=int(self.dim / 8), sn=sn)
        self.ln1 = nn.Linear(self.ssize * self.ssize * 8 * self.dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = input.contiguous()
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = output.view(-1, self.ssize * self.ssize * 8 * self.dim)
        output = self.ln1(output)
        output = self.tanh(output)
        output = output.view(output.shape[0], -1)
        return output


class GoodGenerator128(nn.Module):
    def __init__(self, input_dim=128, dim=128, chn=64):
        super(GoodGenerator128, self).__init__()

        self.chn = chn

        self.ssize = dim // 32
        self.ln1 = nn.Linear(input_dim, self.ssize * self.ssize * 16 * self.chn)
        self.rb1 = ResidualBlock(16 * self.chn, 16 * self.chn, 3, resample='up')
        self.rb2 = ResidualBlock(16 * self.chn, 8 * self.chn, 3, resample='up')
        self.rb3 = ResidualBlock(8 * self.chn, 4 * self.chn, 3, resample='up')
        self.rb4 = ResidualBlock(4 * self.chn, 2 * self.chn, 3, resample='up')
        self.rb5 = ResidualBlock(2 * self.chn, 1 * self.chn, 3, resample='up')
        self.bn = nn.BatchNorm2d(self.chn)

        self.conv1 = MyConvo2d(1 * self.chn, 3, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.ln1(input.contiguous())
        output = output.view(-1, 16 * self.chn, self.ssize, self.ssize)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = self.rb5(output)

        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.tanh(output)
        return output


class GoodDiscriminator128(nn.Module):
    def __init__(self, dim=128, output_dim=128, chn=64, sn=False):
        super(GoodDiscriminator128, self).__init__()

        self.chn = chn

        self.ssize = dim // 32
        self.conv1 = MyConvo2d(3, self.chn, 3, he_init=False, sn=sn) # size: 128 --> 128
        self.rb1 = ResidualBlock(self.chn, 2 * self.chn, 3, resample='down', hw=dim, sn=sn)
        self.rb2 = ResidualBlock(2 * self.chn, 4 * self.chn, 3, resample='down', hw=int(dim / 2), sn=sn)
        self.rb3 = ResidualBlock(4 * self.chn, 8 * self.chn, 3, resample='down', hw=int(dim / 4), sn=sn)
        self.rb4 = ResidualBlock(8 * self.chn, 16 * self.chn, 3, resample='down', hw=int(dim / 8), sn=sn)
        self.rb5 = ResidualBlock(16 * self.chn, 16 * self.chn, 3, resample='down', hw=int(dim / 16), sn=sn)
        self.ln1 = nn.Linear(self.ssize * self.ssize * 16 * self.chn, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = input.contiguous()
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.rb3(output)
        output = self.rb4(output)
        output = self.rb5(output)
        output = output.view(-1, self.ssize * self.ssize * 16 * self.chn)
        output = self.ln1(output)
        output = self.tanh(output)
        output = output.view(output.shape[0], -1)
        return output
