import torch.nn as nn
# import models.basicblock as B
from collections import OrderedDict

"""
# --------------------------------------------
# DnCNN (20 conv layers)
# FDnCNN (20 conv layers)
# IRCNN (7 conv layers)
# --------------------------------------------
# References:
@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}
@article{zhang2018ffdnet,
  title={FFDNet: Toward a fast and flexible solution for CNN-based image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={9},
  pages={4608--4622},
  year={2018},
  publisher={IEEE}
}
# --------------------------------------------
"""


# --------------------------------------------
# DnCNN
# --------------------------------------------
class DnCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=20, act_mode='BR'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias)
        m_body = [conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_tail = conv(nc, out_nc, mode='C', bias=bias)

        self.model = sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        n = self.model(x)
        return x - n


# --------------------------------------------
# IRCNN denoiser
# --------------------------------------------
class IRCNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=64):
        """
        # ------------------------------------
        denoiser of IRCNN
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(IRCNN, self).__init__()
        L = []
        L.append(
            nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=4, dilation=4, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        L.append(nn.ReLU(inplace=True))
        L.append(
            nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        self.model = sequential(*L)

    def forward(self, x):
        n = self.model(x)
        return x - n


# --------------------------------------------
# FDnCNN
# --------------------------------------------
# Compared with DnCNN, FDnCNN has three modifications:
# 1) add noise level map as input
# 2) remove residual learning and BN
# 3) train with L1 loss
# may need more training time, but will not reduce the final PSNR too much.
# --------------------------------------------
class FDnCNN(nn.Module):
    def __init__(self, in_nc=2, out_nc=1, nc=64, nb=20, act_mode='R'):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        """
        super(FDnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias)
        m_body = [conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in range(nb - 2)]
        m_tail = conv(nc, out_nc, mode='C', bias=bias)

        self.model = sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x


def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR',
         negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                          stride=stride,
                          padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


def sequential(*args):
    """Advanced nn.Sequential.
    Args:
        nn.Sequential, nn.Module
    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


if __name__ == '__main__':
    # from utils import utils_model
    import torch

    model1 = DnCNN(in_nc=1, out_nc=1, nc=64, nb=20, act_mode='BR')
    # print(utils_model.describe_model(model1))

    # model2 = FDnCNN(in_nc=2, out_nc=1, nc=64, nb=20, act_mode='R')
    # print(utils_model.describe_model(model2))

    x = torch.randn((1, 1, 240, 240))
    x1 = model1(x)
    print(x1.shape)
    #
    # x = torch.randn((1, 2, 240, 240))
    # x2 = model2(x)
    # print(x2.shape)
