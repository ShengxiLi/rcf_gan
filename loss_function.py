import torch
import torch.nn as nn
import os


def calculate_norm(x_r, x_i):
    return torch.sqrt(torch.mul(x_r, x_r) + torch.mul(x_i, x_i))


def calculate_imag(x):
    return torch.mean(torch.sin(x), dim=1)


def calculate_real(x):
    return torch.mean(torch.cos(x), dim=1)


class CFLossFunc(nn.Module):
    """
    CF loss function in terms of phase and amplitude difference
    Args:
        loss_type: a specification of choosing types of CF loss, we use the original one in this version
        alpha: the weight for amplitude in CF loss, from 0-1
        beta: the weight for phase in CF loss, from 0-1
        threshold: this is mainly used to reduce the effect of CF values around
                    some zero-around t, we do not use this technique in this paper by setting it to 1, you can refer to
                    https://link.springer.com/chapter/10.1007/978-3-030-30487-4_27 for more details
    """
    def __init__(self, loss_type, alpha, beta, threshold):
        super(CFLossFunc, self).__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

    def forward(self, t, x, target):
        t_x = torch.mm(t, x.t())
        t_x_real = calculate_real(t_x)
        t_x_imag = calculate_imag(t_x)
        t_x_norm = calculate_norm(t_x_real, t_x_imag)

        t_target = torch.mm(t, target.t())
        t_target_real = calculate_real(t_target)
        t_target_imag = calculate_imag(t_target)
        t_target_norm = calculate_norm(t_target_real, t_target_imag)

        amp_diff = t_target_norm - t_x_norm
        loss_amp = torch.mul(amp_diff, amp_diff)

        loss_pha = 2 * (torch.mul(t_target_norm, t_x_norm) -
                        torch.mul(t_x_real, t_target_real) -
                        torch.mul(t_x_imag, t_target_imag))

        loss_pha = loss_pha.clamp(min=1e-12)  # keep numerical stability

        if self.loss_type['normalization'] is not None:
            # normalization
            if self.loss_type['normalization'] == 'xx':
                normalization = torch.mul(t_x_norm, t_x_norm)
            elif self.loss_type['normalization'] == 'xy':
                normalization = torch.mul(t_x_norm, t_target_norm)
            elif self.loss_type['normalization'] == 'yy':
                normalization = torch.mul(t_target_norm, t_target_norm)
            elif self.loss_type['normalization'] == 'tt':
                normalization = torch.mean(torch.abs(t), dim=1)
            else:
                raise SystemExit('Error: Unknown normalization type \'{0}\''.format(self.loss_type['normalization']))

            # weight
            if self.loss_type['threshold'] == 'origin':
                normalization = torch.clamp(normalization, min=self.threshold)
            elif self.loss_type['threshold'] == 'relu':
                normalization[normalization < self.threshold] = 2
            else:
                raise SystemExit('Error: Unknown weight type \'{0}\''.format(self.loss_type['weight']))

        if self.loss_type['normalization'] is not None:
            loss = torch.mean(torch.div((self.alpha * loss_amp + self.beta * loss_pha), normalization))
        else:
            loss = torch.mean(torch.sqrt(self.alpha * loss_amp + self.beta * loss_pha))
        return loss
