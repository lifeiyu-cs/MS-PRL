from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms


class Net1(nn.Module):
    def __init__(self, image_channel):
        super(Net1, self).__init__()
        dim = 64
        self.c1 = nn.Sequential(
            nn.Conv2d(image_channel, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.c2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.c3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.c4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim * 4, dim * 8, kernel_size=1, stride=1, padding=0)
        )

        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(dim * 8, dim * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(dim * 4 * 2, dim * 4, kernel_size=1, stride=1, padding=0)
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(dim * 2 * 2, dim * 2, kernel_size=1, stride=1, padding=0)
        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(dim * 2, dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0)
        self.d4 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1, padding=0)
        self.out2 = nn.Conv2d(dim // 2, image_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.c1(x)
        res1 = out
        out = self.c2(out)
        res2 = out
        out = self.c3(out)
        res3 = out
        out = self.c4(out)
        res4 = out
        out = self.out1(out)

        out = self.d1(out)
        out = torch.cat([out, res4], dim=1)
        out = self.conv1(out)
        out = self.d2(out)
        out = torch.cat([out, res3], dim=1)
        out = self.conv2(out)
        out = self.d3(out)
        out = torch.cat([out, res2], dim=1)
        out = self.conv3(out)
        out = self.d4(out)
        out = torch.cat([out, res1], dim=1)
        out = self.conv4(out)

        out = self.out2(out)

        return out


class Net2(nn.Module):
    def __init__(self, image_channel):
        super(Net2, self).__init__()
        dim = 64
        self.c1 = nn.Sequential(
            nn.Conv2d(image_channel, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.c2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.c3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.c4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim * 4, dim * 8, kernel_size=1, stride=1, padding=0)
        )

        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(dim * 8, dim * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(dim * 4 * 2, dim * 4, kernel_size=1, stride=1, padding=0)
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(dim * 2 * 2, dim * 2, kernel_size=1, stride=1, padding=0)
        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(dim * 2, dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0)
        self.d4 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1, padding=0)
        self.out2 = nn.Conv2d(dim // 2, image_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.c1(x)
        res1 = out
        out = self.c2(out)
        res2 = out
        out = self.c3(out)
        res3 = out
        out = self.c4(out)
        res4 = out
        out = self.out1(out)

        out = self.d1(out)
        out = torch.cat([out, res4], dim=1)
        out = self.conv1(out)
        out = self.d2(out)
        out = torch.cat([out, res3], dim=1)
        out = self.conv2(out)
        out = self.d3(out)
        out = torch.cat([out, res2], dim=1)
        out = self.conv3(out)
        out = self.d4(out)
        out = torch.cat([out, res1], dim=1)
        out = self.conv4(out)

        out = self.out2(out)

        return out


class Net3(nn.Module):
    def __init__(self, image_channel):
        super(Net3, self).__init__()
        dim = 64
        self.c1 = nn.Sequential(
            nn.Conv2d(image_channel * 3, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.c2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.c3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.c4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(dim * 4, dim * 8, kernel_size=1, stride=1, padding=0)
        )

        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(dim * 8, dim * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(dim * 4 * 2, dim * 4, kernel_size=1, stride=1, padding=0)
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(dim * 2 * 2, dim * 2, kernel_size=1, stride=1, padding=0)
        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(dim * 2, dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1, padding=0)
        self.d4 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1, padding=0)
        self.out2 = nn.Conv2d(dim // 2, image_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.c1(x)
        res1 = out
        out = self.c2(out)
        res2 = out
        out = self.c3(out)
        res3 = out
        out = self.c4(out)
        res4 = out
        out = self.out1(out)

        out = self.d1(out)
        out = torch.cat([out, res4], dim=1)
        out = self.conv1(out)
        out = self.d2(out)
        out = torch.cat([out, res3], dim=1)
        out = self.conv2(out)
        out = self.d3(out)
        out = torch.cat([out, res2], dim=1)
        out = self.conv3(out)
        out = self.d4(out)
        out = torch.cat([out, res1], dim=1)
        out = self.conv4(out)

        out = self.out2(out)

        return out


class GGRL(nn.Module):
    def __init__(self, image_channel, device='cuda', num_blocks=8):
        super(GGRL, self).__init__()
        dim = 64
        # ------------------------------ Content Aggregation ------------------------------
        self.net1 = Net1(image_channel)
        self.sobel_x = Gradient_Map_X(device)
        self.net2 = Net2(image_channel)
        self.sobel_y = Gradient_Map_Y(device)
        self.net3 = Net3(image_channel)

        self.conv1 = nn.Conv2d(image_channel * 2, dim, kernel_size=3, stride=1, padding=1)
        self.rbg = ResBlockGroup(dim, k=3, num_res=num_blocks)

        self.conv2 = nn.Conv2d(dim, dim * 4, kernel_size=3, stride=1, padding=1)
        self.out = nn.Conv2d(dim * 4, image_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.net1(x)
        sobel_x = self.sobel_x(x1)
        x2 = self.net2(x)
        sobel_y = self.sobel_y(x2)
        x3 = torch.cat([sobel_x, sobel_y, x], dim=1)
        x3 = self.net3(x3)

        out = torch.cat([x, x3], dim=1)
        out = self.conv1(out)
        out = self.rbg(out)
        out = self.conv2(out)
        out = self.out(out)
        out = out + x3
        out = torch.tanh(out)
        return out


class ResBlockGroup(nn.Module):
    def __init__(self, channel, k, num_res):
        super(ResBlockGroup, self).__init__()

        layers = [ResBlock(channel, channel, k=k, planes=channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, k, planes):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, planes, kernel_size=k, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, out_channel, kernel_size=k, padding=1, stride=1)
        )

    def forward(self, x):
        res = self.conv(x)
        res = res + x
        return res


class Gradient_Map(nn.Module):
    def __init__(self, device):
        super(Gradient_Map, self).__init__()
        self.sobel_filter_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).reshape((1, 1, 3, 3))
        self.sobel_filter_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).reshape((1, 1, 3, 3))
        self.sobel_filter_X = torch.from_numpy(self.sobel_filter_X).float().to(device)
        self.sobel_filter_Y = torch.from_numpy(self.sobel_filter_Y).float().to(device)
        self.act = nn.Sigmoid()

    def forward(self, output):
        b, c, h, w = output.size()

        output_X_c, output_Y_c = [], []
        for i in range(c):
            output_grad_X = F.conv2d(output[:, i:i + 1, :, :], self.sobel_filter_X, bias=None, stride=1, padding=1)
            output_grad_Y = F.conv2d(output[:, i:i + 1, :, :], self.sobel_filter_Y, bias=None, stride=1, padding=1)

            output_X_c.append(output_grad_X)
            output_Y_c.append(output_grad_Y)

        output_X = torch.cat(output_X_c, dim=1)
        output_Y = torch.cat(output_Y_c, dim=1)

        output_X = self.act(output_X)
        output_Y = self.act(output_Y)

        return output_X, output_Y


class Gradient_Map_X(nn.Module):
    def __init__(self, device):
        super(Gradient_Map_X, self).__init__()
        self.sobel_filter_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).reshape((1, 1, 3, 3))
        self.sobel_filter_X = torch.from_numpy(self.sobel_filter_X).float().to(device)
        self.act = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        output_X_c = []
        for i in range(c):
            output_grad_X = F.conv2d(x[:, i:i + 1, :, :], self.sobel_filter_X, bias=None, stride=1, padding=1)
            output_X_c.append(output_grad_X)

        output_X = torch.cat(output_X_c, dim=1)
        output_X = self.act(output_X)
        return output_X


class Gradient_Map_Y(nn.Module):
    def __init__(self, device):
        super(Gradient_Map_Y, self).__init__()
        self.sobel_filter_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).reshape((1, 1, 3, 3))
        self.sobel_filter_Y = torch.from_numpy(self.sobel_filter_Y).float().to(device)
        self.act = nn.Sigmoid()

    def forward(self, y):
        b, c, h, w = y.size()
        output_Y_c = []
        for i in range(c):
            output_grad_Y = F.conv2d(y[:, i:i + 1, :, :], self.sobel_filter_Y, bias=None, stride=1, padding=1)
            output_Y_c.append(output_grad_Y)

        output_Y = torch.cat(output_Y_c, dim=1)
        output_Y = self.act(output_Y)
        return output_Y


if __name__ == '__main__':
    net = GGRL(image_channel=1, device='cpu')
    # print(net)
    y = net(torch.randn(16, 1, 128, 128))
    print(y.size())
