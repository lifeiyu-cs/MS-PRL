from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class PRL(nn.Module):
    def __init__(self, image_channel, num_blocks=[2, 4, 2, 8]):
        super(PRL, self).__init__()
        dim = 64
        # ------------------------------ Content Aggregation ------------------------------
        self.k7n64s1 = nn.Conv2d(image_channel, dim, kernel_size=3, stride=1, padding=1)

        self.ResBlockGroup1 = ResBlockGroup(dim, k=3, num_res=num_blocks[0])

        self.down1_1_k3n128s2 = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.down1_2_k3n128s1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.down2_1_k3n256s2 = nn.Conv2d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1)
        self.down2_2_k3n256s1 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.ResBlockGroup2 = ResBlockGroup(dim * 4, k=3, num_res=num_blocks[1])

        self.up1_1_k3n128s1 = nn.Conv2d(dim * 4, dim * 2, kernel_size=3, stride=1, padding=1)
        self.up1_2_k3n128s1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up2_1_k3n64s1 = nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1)
        self.up2_2_k3n64s1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.ResBlockGroup3 = ResBlockGroup(dim, k=3, num_res=num_blocks[2])
        self.latent = nn.Conv2d(dim, image_channel, kernel_size=3, stride=1, padding=1)

        self.n64s1_c = nn.Conv2d(2, dim, kernel_size=3, stride=1, padding=1)
        self.ResBlockGroup4 = ResBlockGroup(dim, k=3, num_res=num_blocks[3])
        self.n256s1_2 = nn.Conv2d(dim, dim * 4, kernel_size=3, stride=1, padding=1)

        self.out = nn.Conv2d(dim * 4, image_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # ------------------------------ Content Aggregation ------------------------------
        out = self.k7n64s1(x)

        # 2X residual blocks
        out = self.ResBlockGroup1(out)

        # second residual
        res1 = out
        out = self.down1_1_k3n128s2(out)
        out = self.down1_2_k3n128s1(out)
        res2 = out

        out = self.down2_1_k3n256s2(out)
        out = self.down2_2_k3n256s1(out)

        # 4X residual blocks
        out = self.ResBlockGroup2(out)

        # first image upscale
        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.up1_1_k3n128s1(out)
        out = self.up1_2_k3n128s1(out)
        out = out + res2

        # second image upscale
        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.up2_1_k3n64s1(out)
        out = self.up2_2_k3n64s1(out)
        out = out + res1

        # 2x residual blocks
        out = self.ResBlockGroup3(out)

        out = self.latent(out)
        content_map = out

        # image fussion
        fussion = torch.cat([content_map, x], dim=1)

        # ------------------------------ Detail Generation ------------------------------
        out = self.n64s1_c(fussion)
        # 8x residual blocks
        out = self.ResBlockGroup4(out)

        out = self.n256s1_2(out)
        detail_map = self.out(out)

        output_map = content_map + detail_map
        output_map = torch.tanh(output_map)
        return output_map


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


if __name__ == '__main__':
    net = PRL(image_channel=1)
    # print(net)
    y = net(torch.randn(1, 1, 256, 256))
    print(y.size())
