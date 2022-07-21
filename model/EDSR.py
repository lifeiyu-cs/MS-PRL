import torch
import torch.nn as nn
import math


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean)

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False


class ResBlockGroup(nn.Module):
    def __init__(self, channel, k, num_res):
        super(ResBlockGroup, self).__init__()

        layers = [ResBlock(channel, channel, k=k, planes=channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, k, planes, res_scale=0.1):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, planes, kernel_size=k, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, out_channel, kernel_size=k, padding=1, stride=1)
        )
        self.res_scale = res_scale

    def forward(self, x):
        res = self.conv(x)
        res = res * self.res_scale
        res = res + x
        return res


class _Residual_Block(nn.Module):
    def __init__(self, ):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output *= 0.1
        output = torch.add(output, identity_data)
        return output


class EDSR(nn.Module):
    def __init__(self, image_channel):
        super(EDSR, self).__init__()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.sub_mean = MeanShift(rgb_mean, -1)
        dim = 64
        self.input = nn.Conv2d(in_channels=image_channel, out_channels=dim, kernel_size=3, stride=1, padding=1)

        self.body = ResBlockGroup(channel=dim, k=3, num_res=16)

        self.mid = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)

        # self.upscale4x = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=256 * 4, kernel_size=3, stride=1, padding=1),
        #     nn.PixelShuffle(2),
        #     nn.Conv2d(in_channels=256, out_channels=256 * 4, kernel_size=3, stride=1, padding=1),
        #     nn.PixelShuffle(2),
        # )

        self.output = nn.Conv2d(in_channels=dim, out_channels=image_channel, kernel_size=3, stride=1, padding=1)

        self.add_mean = MeanShift(rgb_mean, 1)

    def forward(self, x):
        # out = self.sub_mean(x)
        out = self.input(x)
        res = out
        out = self.mid(self.body(out))
        out = out + res
        # out = self.upscale4x(out)
        out = self.output(out)
        # out = self.add_mean(out)
        return out


if __name__ == '__main__':
    net = EDSR(image_channel=3)
    # print(net)
    y = net(torch.randn(16, 3, 128, 128))
    print(y.size())
