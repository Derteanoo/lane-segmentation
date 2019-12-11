import torch
import torch.nn as nn
import pdb
from unet import UNet
#from thop import profile
model = UNet()
#flops, params = profile(model, input_size=(1, 3, 256,256))



from torchsummaryX import summary
summary(model, torch.zeros((1, 3, 128, 128)))