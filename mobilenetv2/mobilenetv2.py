import torch.nn as nn
import math
import torch.nn.functional as F
import pdb


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


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

        self.layer = conv_1x1_bn(in_ch, out_ch)

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
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, class_num=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        #self.classifier = nn.Linear(self.last_channel, class_num)

        self._initialize_weights()

        #upsample
        self.src_up1 = up(int(interverted_residual_setting[-1][1] * width_mult), int(interverted_residual_setting[4][1] * width_mult))
        self.src_up2 = up(int(interverted_residual_setting[4][1] * width_mult), int(interverted_residual_setting[2][1] * width_mult))
        self.src_up3 = up(int(interverted_residual_setting[2][1] * width_mult), int(interverted_residual_setting[1][1] * width_mult))
        self.src_up4 = up(int(interverted_residual_setting[1][1] * width_mult), int(interverted_residual_setting[0][1] * width_mult))
        self.src_up5 = last_up(int(interverted_residual_setting[0][1] * width_mult))

        self.src_out = outconv(int(interverted_residual_setting[0][1] * width_mult), class_num)


    def forward(self, x):
        x0 = self.features[0:2](x)#16,16,128,256
        x1 = self.features[2:4](x0)#[16, 24, 64, 128]
        x2 = self.features[4:7](x1)#[16, 32, 32, 64]
        x3 = self.features[7:14](x2)#[16, 96, 16, 32]
        x4 = self.features[14:-1](x3)#[16, 320, 8, 16]

        src_x = self.src_up1(x4,x3)
        src_x = self.src_up2(src_x,x2)
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


def mobilenet_v2(pretrained=True, width_mult=0.25, **kwargs):
    model = MobileNetV2(width_mult=width_mult, **kwargs)

    if pretrained:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url(
            'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True)
        model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    net = mobilenet_v2(True)





