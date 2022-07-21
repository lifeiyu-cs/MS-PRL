import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, k):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=k, padding=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class BlockGroup(nn.Module):
    def __init__(self, channel, k, num_block):
        super(BlockGroup, self).__init__()

        layers = [Block(in_channel=channel, out_channel=channel, k=k) for _ in range(num_block)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class VDSR(nn.Module):
    def __init__(self, image_channel):
        super(VDSR, self).__init__()
        self.body = BlockGroup(channel=64, k=3, num_block=18)
        self.input = nn.Sequential(
            nn.Conv2d(in_channels=image_channel, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.output = nn.Conv2d(in_channels=64, out_channels=image_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.input(x)
        out = self.body(out)
        out = self.output(out)
        out = out + x
        return out


if __name__ == '__main__':
    net = VDSR(image_channel=1)
    # print(net)
    y = net(torch.randn(16, 1, 128, 128))
    print(y.size())
