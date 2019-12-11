# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *


#from dropblock import DropBlock2D, LinearScheduler


class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out, dropout=1):
        super(ResidualBlock, self).__init__()
        layers=[
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True)]
        self.res_pre=nn.Sequential(*layers)

        self.dropblock = None
        if dropout:
            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=0.10, block_size=7),
                start_value=0.0,
                stop_value=0.10,
                nr_steps=5e3
            )

        layers=[
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True)]

        self.res_after=nn.Sequential(*layers)

    def forward(self, x):
        if self.dropblock!=None:
            self.dropblock.step()  # increment number of iterations
            x = self.dropblock(x)
        temp=x
        x=self.res_pre(x)
        x=self.res_after(x)

        return temp + x


class UNet2(nn.Module):
    def __init__(self, in_channel=8, n_classes=5):
        super(UNet2, self).__init__()
        self.src_inc = inconv(3, in_channel)

        self.src_down1 = down(in_channel, in_channel*2)
        in_channel=in_channel * 2  #16
        self.src_down2 = down(in_channel, in_channel * 2)
        in_channel = in_channel * 2 #32
        self.src_down3 = down(in_channel, in_channel * 2)
        in_channel = in_channel * 2  #64
        self.src_down4 = down(in_channel, in_channel * 2)
        in_channel = in_channel * 2  #128
        self.src_down5 = down(in_channel, in_channel * 2)
        in_channel = in_channel * 2  #256

        self.src_up1 = up(in_channel+in_channel//2, in_channel//2)
        in_channel=in_channel//2 #128
        self.src_up2 = up(in_channel+in_channel//2, in_channel // 2)
        in_channel = in_channel // 2  # 64
        self.src_up3 = up(in_channel+in_channel//2, in_channel // 2)
        in_channel = in_channel // 2  # 32
        self.src_up4 = up(in_channel+in_channel//2, in_channel // 2)
        in_channel = in_channel // 2  # 16
        self.src_up5 = up(in_channel+in_channel//2, in_channel // 2)
        in_channel = in_channel // 2  # 8

        self.src_outc = outconv(in_channel, n_classes)

    def forward(self, src):
        src1 = self.src_inc(src)#8

        src2 = self.src_down1(src1)#16
        src3 = self.src_down2(src2)#32
        src4 = self.src_down3(src3)#64
        src5 = self.src_down4(src4)#128
        src6 = self.src_down5(src5)#256

        src_x = self.src_up1(src6,src5)
        src_x = self.src_up2(src_x,src4)
        src_x = self.src_up3(src_x, src3)
        src_x = self.src_up4(src_x, src2)
        src_x = self.src_up5(src_x, src1)

        src_out = self.src_outc(src_x)
        return src_out

class UNet(nn.Module):
    def __init__(self, in_channel=3, class_num = 4):
        super(UNet, self).__init__()

        stride=3
        self.src_inc = inconv(3, in_channel)

        self.src_down1 = down(in_channel, in_channel+stride)
        in_channel=in_channel+stride  #9
        self.src_down2 = down(in_channel, in_channel+stride)
        in_channel = in_channel+stride #12
        self.src_down3 = down(in_channel, in_channel+stride)
        in_channel = in_channel+stride  #20

        stride=8
        self.src_down4 = down(in_channel, in_channel+stride)
        in_channel = in_channel+stride  #28
        self.src_down5 = down(in_channel, in_channel+stride)
        in_channel = in_channel+stride  #36


        self.src_up1 = up(in_channel+in_channel-stride, in_channel-stride)
        in_channel=in_channel-stride #28
        self.src_up2 = up(in_channel+in_channel-stride, in_channel-stride)
        in_channel=in_channel-stride   # 32

        stride=3
        self.src_up3 = up(in_channel+in_channel-stride, in_channel-stride)
        in_channel=in_channel-stride   # 24
        self.src_up4 = up(in_channel+in_channel-stride, in_channel-stride)
        in_channel=in_channel-stride   # 16
        self.src_up5 = up(in_channel+in_channel-stride, in_channel-stride)
        in_channel=in_channel-stride   # 8

        self.src_out = outconv(in_channel, class_num)


    def forward(self, src):
        src1 = self.src_inc(src)#8

        src2 = self.src_down1(src1)#16
        src3 = self.src_down2(src2)#32
        src4 = self.src_down3(src3)#64
        src5 = self.src_down4(src4)#128
        src6 = self.src_down5(src5)#256

        src_x = self.src_up1(src6,src5)
        src_x = self.src_up2(src_x,src4)
        src_x = self.src_up3(src_x, src3)
        src_x = self.src_up4(src_x, src2)
        src_x = self.src_up5(src_x, src1)

        src_out = self.src_out(src_x)
        return src_out