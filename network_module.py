from utility import get_input_white_noise
import math
from ResNet import *


class SampleNet(nn.Module):
    """
    generator net
    Args:
        input_dim: input dimension of Gaussian noise
        target_dim: image size
        img_channel: number of image channels, 1 for mnist and 3 for others
        net_structure: a dictionary that contains necessary parameters for the generator net
    """
    def __init__(self, input_dim, target_dim, img_channel, net_structure):
        super(SampleNet, self).__init__()
        self.layers_list = nn.ModuleList()
        self.cnn_flag = False
        activations = activations_to_torch(net_structure['activations'])
        self.net_type = net_structure['net_type']

        # DCGAN
        if net_structure['net_type'] == 'dcgan':
            self.cnn_flag = True
            ch_in = input_dim
            tt_layer_number = int(math.log2(target_dim) - 1)
            top_channel_number = 64 * 2 ** (tt_layer_number - 2)

            conv_layer = nn.ConvTranspose2d(ch_in, top_channel_number, 4, 1, 0, bias=False)
            self.layers_list.append(conv_layer)
            bn_layer = nn.BatchNorm2d(top_channel_number)
            self.layers_list.append(bn_layer)
            self.layers_list.append(activations[0])

            ch_in = top_channel_number
            ch_out = top_channel_number
            for i in range(1, tt_layer_number - 1):
                ch_out = ch_out // 2
                conv_layer = nn.ConvTranspose2d(ch_in, ch_out, 4, 2, 1, bias=False)
                self.layers_list.append(conv_layer)
                bn_layer = nn.BatchNorm2d(ch_out)
                self.layers_list.append(bn_layer)
                self.layers_list.append(activations[0])
                ch_in = ch_out

            conv_layer = nn.ConvTranspose2d(ch_in, img_channel, 4, 2, 1, bias=False)
            self.layers_list.append(conv_layer)
            self.layers_list.append(activations[1])
        elif self.net_type == 'resnet':
            self.cnn_flag = True
            if target_dim == 64:
                self.model_gen = GoodGenerator64(input_dim, target_dim)
            elif target_dim == 128:
                self.model_gen = GoodGenerator128(input_dim, target_dim)
            else:
                Exception('Invalid target dim. Please define the net structure by yourself!')

    def forward(self, noise):
        a = noise
        if self.net_type == 'resnet':
            a = self.model_gen(a)
        else:
            if self.cnn_flag:
                a = a.unsqueeze(-1).unsqueeze(-1)
            for layer in self.layers_list:
                a = layer(a)
        return a


class AdvNet(nn.Module):
    """
    critic net
    Args:
        channel_in: number of image channels, 1 for mnist and 3 for others
        target_dim: image size
        feature_dim: dimension of embedded vectors, equal to the input dimension of Gaussian noise to the generator
        net_structure: a dictionary that contains necessary parameters for the critic net
    """
    def __init__(self, channel_in, target_dim, feature_dim, net_structure):
        super(AdvNet, self).__init__()
        self.cnn_flag = False
        self._input_adv_t_net_dim = feature_dim
        self.channel_in = channel_in
        self.t_sigma_num = net_structure['adv_t_sigma_num']
        self.net_type = net_structure['net_type']
        activations = activations_to_torch(net_structure['activations_a'])

        # DCGAN
        if net_structure['net_type'] == 'dcgan':
            self.cnn_flag = True
            self.convlayers_list = nn.ModuleList()
            tt_layer_number = int(math.log2(target_dim) - 1)
            ch_in = self.channel_in

            out_channel_number = 64
            conv_layer = nn.Conv2d(ch_in, out_channel_number, 4, 2, 1, bias=False)
            self.convlayers_list.append(conv_layer)
            self.convlayers_list.append(activations[0])
            ch_in = out_channel_number
            ch_out = out_channel_number
            for i in range(1, tt_layer_number - 1):
                ch_out = ch_out * 2
                conv_layer = nn.Conv2d(ch_in, ch_out, 4, 2, 1, bias=False)
                self.convlayers_list.append(conv_layer)
                bn_layer = nn.LayerNorm(
                    [ch_out, target_dim // 2 ** (i + 1), target_dim // 2 ** (i + 1)])
                # if you wish to use batch normalisation
                # bn_layer = nn.BatchNorm2d(ch_out)
                self.convlayers_list.append(bn_layer)
                # if you wish to use spectral normalisation
                # conv_sn_layer = nn.utils.spectral_norm(nn.Conv2d(ch_in, ch_out, 4, 2, 1, bias=False))
                # self.convlayers_list.append(conv_sn_layer)
                self.convlayers_list.append(activations[0])
                ch_in = ch_out
            conv_layer = nn.Conv2d(ch_in, feature_dim, 4, 1, 0, bias=False)
            self.convlayers_list.append(conv_layer)
            bn_layer = nn.LayerNorm([feature_dim, 1, 1])
            self.convlayers_list.append(bn_layer)
            self.convlayers_list.append(activations[1])
        elif self.net_type == 'resnet':
            if target_dim == 64:
                self.model_adv = GoodDiscriminator64(target_dim, feature_dim, sn=True)
            elif target_dim == 128:
                self.model_adv = GoodDiscriminator128(target_dim, feature_dim, sn=True)
            else:
                Exception('Invalid target dim. Please define the net structure by yourself!')

        if self.t_sigma_num > 0:
            # adversarial nets for t scales
            self.t_layers_list = nn.ModuleList()
            ch_in = feature_dim
            activations = activations_to_torch(net_structure['activations_t'])
            # a simple 3-layer fc net
            for i in range(3):
                self.t_layers_list.append(nn.Linear(ch_in, ch_in))
                self.t_layers_list.append(nn.BatchNorm1d(ch_in))
                activation = activations[0] if i < 2 else activations[1]
                self.t_layers_list.append(activation)

        # base mean and covariance
        self._input_t_dim = net_structure['input_t_dim']
        self._input_t_batchsize = net_structure['input_t_batchsize']
        self._input_t_var = net_structure['input_t_var']

    def forward(self, noise):
        a = noise
        if self.net_type == 'resnet':
            a = self.model_adv(a)
            return a
        else:
            for layer in self.convlayers_list:
                a = layer(a)
            a = a.view(a.shape[0], -1)
            return a

    def net_t(self):
        if self.t_sigma_num > 0:
            # use t_net
            device = 'cpu'
            if torch.cuda.is_available():
                device = torch.device('cuda')
            self._t_net_input = get_input_white_noise(self._input_adv_t_net_dim, self._input_t_var,
                                                      self.t_sigma_num).detach().to(device)
            a = self._t_net_input
            for layer in self.t_layers_list:
                a = layer(a)
            a = a.repeat(int(self._input_t_batchsize / self.t_sigma_num), 1)
            self._t = get_input_white_noise(self._input_t_dim, self._input_t_var / self._input_t_dim,
                                            # dimension normalisation
                                            self._input_t_batchsize).detach().to(device)
            self._t = self._t * a
        else:
            # use fixed Gaussian
            self._t = get_input_white_noise(self._input_t_dim, self._input_t_var / self._input_t_dim,
                                            self._input_t_batchsize).detach()
            device = 'cpu'
            if torch.cuda.is_available():
                device = torch.device('cuda')
            self._t = self._t.to(device)
            # add incremental functions here
        return self._t


def activations_to_torch(activations):
    for i, i_act in enumerate(activations):
        if i_act[0] == 'tanh':
            activations[i] = nn.Tanh()
        elif i_act[0] == 'sigmoid':
            activations[i] = nn.Sigmoid()
        elif i_act[0] == 'relu':
            activations[i] = nn.ReLU()
        elif i_act[0] == 'lrelu':
            activations[i] = nn.LeakyReLU(negative_slope=i_act[1])
        elif i_act[0] is None:
            pass
        else:
            raise SystemExit('Error: Unknown activation function \'{0}\''.format(i_act[0]))
    return activations
