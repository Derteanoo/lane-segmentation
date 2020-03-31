from torch.autograd import Variable
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
import pdb


def Conv_3x3(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def Conv_1x1(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def SepConv_3x3(inp, oup): #input=32, output=16
    return nn.Sequential(
        # dw
        nn.Conv2d(inp, inp , 3, 1, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )


class double_conv(nn.Module):
    '''(conv => IN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class last_up(nn.Module):
    def __init__(self, in_ch):
        super(last_up, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')  # , align_corners=True)

        self.conv = double_conv(in_ch, in_ch)

    def forward(self, x1):
        x1 = self.up(x1)
        x = self.conv(x1)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')  # , align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch-out_ch, in_ch-out_ch, 2, stride=2)

        self.conv = double_conv(out_ch, out_ch)

        self.layer = Conv_1x1(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.layer(x1)
        x = x2+x1
        
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        #self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        #x = self.tanh(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, kernel):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel, stride, kernel // 2, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MnasNet(nn.Module):
    def __init__(self, class_num=1000, input_size=224, width_mult=1.):
        super(MnasNet, self).__init__()

        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s, k
            [3, 24,  3, 2, 3],  # -> 56x56
            [3, 40,  3, 2, 5],  # -> 28x28
            [6, 80,  3, 2, 5],  # -> 14x14
            [6, 96,  2, 1, 3],  # -> 14x14
            [6, 192, 4, 2, 5],  # -> 7x7
            [6, 320, 1, 1, 3],  # -> 7x7
        ]

        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        # building first two layer
        self.features = [Conv_3x3(3, input_channel, 2), SepConv_3x3(input_channel, 16)]
        input_channel = 16

        # building inverted residual blocks (MBConv)
        for t, c, n, s, k in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t, k))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t, k))
                input_channel = output_channel

        # building last several layers
        self.features.append(Conv_1x1(input_channel, self.last_channel))
        self.features.append(nn.AdaptiveAvgPool2d(1))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        #self.classifier = nn.Sequential(
        #    nn.Dropout(),
        #    nn.Linear(self.last_channel, n_class),
        #)

        self._initialize_weights()

        #upsample
        self.src_up1 = up(self.last_channel, int(self.interverted_residual_setting[3][1] * width_mult))
        self.src_up2 = up(int(self.interverted_residual_setting[3][1] * width_mult), int(self.interverted_residual_setting[1][1] * width_mult))
        self.src_up3 = up(int(self.interverted_residual_setting[1][1] * width_mult), int(self.interverted_residual_setting[0][1] * width_mult))
        self.src_up4 = up(int(self.interverted_residual_setting[0][1] * width_mult), 16)
        self.src_up5 = last_up(16)

        self.src_out = outconv(16, class_num)

    def forward(self, x):
        x0 = self.features[0:2](x)#32, 16, 128, 256
        x1 = self.features[2:5](x0)#([32, 24, 64, 128])
        x2 = self.features[5:8](x1)#[32, 40, 32, 64]
        x3 = self.features[8:13](x2)#([32, 96, 16, 32])
        x4 = self.features[13:-1](x3)#[32, 1280, 8, 16]

        src_x = self.src_up1(x4, x3)
        src_x = self.src_up2(src_x, x2)
        src_x = self.src_up3(src_x, x1)
        src_x = self.src_up4(src_x, x0)
        src_x = self.src_up5(src_x)

        src_out = self.src_out(src_x)

        return src_out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    net = MnasNet()
    x_image = Variable(torch.randn(1, 3, 224, 224))
    y = net(x_image)
    # print(y)




