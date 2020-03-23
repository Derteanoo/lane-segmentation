#########################################################################
##
## Structure of network.
##
#########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util_hourglass import *

####################################################################
##
## lane_detection_network
##
####################################################################
class Hourglass(nn.Module):
    def __init__(self, class_num=5):
        super(Hourglass, self).__init__()

        self.resizing = resize_layer(3, 128)

        #feature extraction
        self.layer1 = hourglass_block(128, 128)
        #self.layer2 = hourglass_block(128, 128)
        self.last_layer =  nn.Conv2d(
                in_channels=128,
                out_channels=class_num,
                kernel_size=3,
                stride=1,
                padding=1)

    def forward(self, inputs):
        h = inputs.size(2)
        w = inputs.size(3)
        
        #feature extraction
        out = self.resizing(inputs)#[8,128,32,64]
        
        #result1, out = self.layer1(out)#3*[8,1,32,64],3*[8,2,32,64],3*[8,4,32,64]
        #result2, out = self.layer2(out)#3*[8,1,32,64],3*[8,2,32,64],3*[8,4,32,64]
        out = self.layer1(out)

        out = self.last_layer(out)

        out = F.upsample(input=out, size=(h, w), mode='bilinear')

        return out
