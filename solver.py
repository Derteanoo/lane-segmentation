import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import Variable
from torchvision.utils import save_image
#from model import Discriminator,vgg16
#from model import Generator as Generator, Generator_orig
#from unet import UNet as Generator
from PIL import Image
from collections import OrderedDict
from scipy.ndimage.filters import gaussian_filter

from torch.optim.lr_scheduler import MultiStepLR

from loss import SoftmaxFocalLoss
from loss import SoftDiceLoss

import pdb

class Solver(object):

    def __init__(self, data_loader, data_loader2, config):
        # Data loader
        self.data_loader = data_loader
        self.data_loader2 = data_loader2
        self.class_num = config.class_num

        # Model hyper-parameters
        self.d_train_repeat = config.d_train_repeat

        # Hyper-parameteres
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp
        self.g_lr = config.g_lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.batch_size = config.batch_size
        self.use_tensorboard = config.use_tensorboard
        self.pretrained_model = config.pretrained_model
        self.loss = config.loss

        # Test settings
        self.test_model = config.test_model

        # Path
        self.log_path = config.log_path
        self.sample_path = config.sample_path
        self.model_save_path = config.model_save_path
        self.result_path = config.result_path

        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.val_log_step = config.val_log_step

        # Build tensorboard if use
        self.build_model(config.network, config.class_num, config.print_network)
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self, network, class_num, print_network):
        # Define a generator and a discriminator
        if network == 'unet':
            from unet import UNet as Generator
        elif network == 'deeplabv3':
            from deeplabv3 import DeepLabV3 as Generator
        elif network == 'espnetv2':
            from espnetv2 import EESPNet_Seg as Generator
        elif network == 'enet':
            from enet import ENet as Generator
        elif network == 'erfnet':
            from erfnet import ERFNet as Generator
        elif network == 'hrnet':
            from hrnet import HRNetV2 as Generator

        self.G = Generator(class_num = class_num)

        #self.D = Discriminator()
        #self.vgg = vgg16()

        self.optimizer_g = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        #self.optimizer_d = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        if print_network == True:
            self.print_network(self.G, 'G')

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.G = nn.DataParallel(self.G).cuda()
        elif torch.cuda.is_available():
            self.G = self.G.cuda()
            #from torchsummary import summary
            #summary(self.G, (3, 256, 256))

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):

        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))),strict=False)

        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, g_lr):
        for param_group in self.optimizer_g.param_groups:
            param_group['lr'] = g_lr


    def reset_grad(self):
        self.optimizer_g.zero_grad()


    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def denorm_mask(self, x):
        #out = (x + 1) / 2
        return x.clamp_(0, 1)

    def vgg_loss(self,x,y):
        loss=0
        for a,b in zip(x,y):
            #loss+=torch.mean(torch.pow(a - b, 2))
            loss += torch.mean(torch.abs(a - b))
        return loss

    def train(self):
        scheduler = MultiStepLR(self.optimizer_g, milestones=[int(x) for x in '10,20,30,200'.split(',')])
        if self.loss == 'nll':
            class_weight = np.asarray([0.1,  # background
                                1,  # solid
                                1.5,  # broken
                                1.5,
                                1], np.float32)  #fishbone
            class_weight = class_weight / np.sum(class_weight)
            class_weight = torch.from_numpy(class_weight)
            seg_criterion = nn.NLLLoss2d(class_weight).cuda()
        elif self.loss == 'focal':
            seg_criterion = SoftmaxFocalLoss(gamma=2,OHEM_percent=0.05).cuda()
        elif self.loss == 'dice':
            class_weight = np.asarray([0.1,  # background
                                1,  # solid
                                1.5,  # broken
                                1.5,
                                1], np.float32)  #fishbone
            class_weight = class_weight / np.sum(class_weight)
            class_weight = torch.from_numpy(class_weight)
            seg_criterion = SoftDiceLoss().cuda()

        """Train StarGAN within a single dataset."""

        # The number of iterations per epoch
        iters_per_epoch = len(self.data_loader)

        fixed_A = []
        fixed_B = []

        for i, (A, B) in enumerate(self.data_loader):
            fixed_A.append(A)
            fixed_B.append(B)
            if i == 1:
                break

        # Fixed inputs and target domain labels for debugging
        fixed_A = torch.cat(fixed_A, dim=0)
        fixed_A = self.to_var(fixed_A, volatile=True)

        fixed_B = torch.cat(fixed_B, dim=0)
        fixed_B = self.to_var(fixed_B, volatile=True)
        fixed_B = fixed_B.unsqueeze(1)
        fixed_B = torch.cat([fixed_B, fixed_B, fixed_B], 1).float()

        fake_image_list = [fixed_A, 2*(fixed_B.float()/(self.class_num-1)-0.5)]
        fake_images = torch.cat(fake_image_list, dim=3)
        save_image(self.denorm(fake_images.data.cpu()),
                   os.path.join(self.sample_path, 'target.png'), nrow=1, padding=0)
        print('Translated images and saved into {}..!'.format(self.sample_path))

        # lr cache for decaying
        g_lr = self.g_lr
        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 0

        # Start training
        start_time = time.time()

        for e in range(start, self.num_epochs):
            for i, (real_A,real_B) in enumerate(self.data_loader):
                real_A = self.to_var(real_A)
                real_B = self.to_var(real_B)

                if real_A.shape[0] != self.batch_size:
                    break

                if (i + 1) % self.sample_step == 0:
                    with torch.no_grad():
                        self.G.eval()
                        fake_image_list = [fixed_A]
                        fake_B=self.G(fixed_A)

                        _, out_mask = torch.max(fake_B, dim=1)
                        out_mask = out_mask.unsqueeze(1)
                        out_mask = torch.cat([out_mask, out_mask, out_mask], 1).float()
                        fake_image_list.append(2 * (out_mask / (self.class_num - 1) - 0.5))
                        fake_image_list.append(2*(fixed_B/(self.class_num - 1) - 0.5))

                        fake_images = torch.cat(fake_image_list, dim=3)
                        save_image(self.denorm(fake_images.data.cpu()),
                               os.path.join(self.sample_path, '{}_{}_fake.png'.format(e + 1, i + 1)), nrow=1, padding=0)
                        print('Translated images and saved into {}..!'.format(self.sample_path))

                        self.G.train()

                # for a(day)
                out_mask=self.G(real_A)
                if self.loss == 'nll' or self.loss == 'focal':
                    total_loss = seg_criterion(F.log_softmax(out_mask, dim=1), real_B)
                elif self.loss == 'dice':
                    onehot_B = F.one_hot(real_B, self.class_num).permute(0,3,1,2)
                    total_loss = seg_criterion(F.log_softmax(out_mask, dim=1), onehot_B, class_weight)

                self.reset_grad()
                total_loss.backward()
                self.optimizer_g.step()

                # Logging
                loss={}
                loss['total_loss'] = total_loss.item()
                
                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                        elapsed, e+1, self.num_epochs, i+1, iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)

                    print(log)
                    
                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)


                # Save model checkpoints
                if (i+1) % self.model_save_step == 0:
                    torch.save(self.G.state_dict(),
                               os.path.join(self.model_save_path, '{}_{}_G.pth'.format(e + 1, i + 1)))
            
            if (e + 1) % self.val_log_step == 0:
                val_iters_per_epoch = len(self.data_loader2)

                for i, (real_A,real_B) in enumerate(self.data_loader2):
                    real_A = self.to_var(real_A)
                    real_B = self.to_var(real_B)

                    if real_A.shape[0] != self.batch_size:
                        break

                    # for a(day)
                    tgt_out=self.G(real_A)
                    if self.loss == 'nll' or self.loss == 'focal':
                        val_loss = seg_criterion(F.log_softmax(tgt_out, dim=1), real_B)
                    elif self.loss == 'dice':
                        onehot_B = F.one_hot(real_B).permute(0,3,1,2)
                        val_loss = seg_criterion(F.log_softmax(tgt_out, dim=1), onehot_B, class_weight)

                    # Logging
                    loss={}
                    loss['val_loss'] = val_loss.item()

                    # Print out log info
                    log = "Epoch [{}/{}], val_Iter [{}/{}]".format(
                        e+1, self.num_epochs, i+1, val_iters_per_epoch)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)

                    print(log)
                    
                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * val_iters_per_epoch + i + 1)

            # Decay learning rate
            if (e+1) > (self.num_epochs - self.num_epochs_decay):
                g_lr -= (self.g_lr / float(self.num_epochs_decay))
                self.update_lr(g_lr)
                print ('Decay learning rate to g_lr: {},.'.format(g_lr))


    def test(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))

        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()

        start = time.time()
        
        with torch.no_grad():
            for i, (real_A, real_B) in enumerate(self.data_loader):
                print('i=',i)
                real_A = self.to_var(real_A)
                real_B = self.to_var(real_B)

                fake_image_list = []
                fake_B = self.G(real_A)
                upsample = nn.Upsample(size=(real_B.shape[2]//2,real_B.shape[3]),mode='nearest')
                fake_B = upsample(fake_B)
                _, out_mask = torch.max(fake_B, dim=1)
                out_mask = out_mask.unsqueeze(1)
                out_mask = torch.cat([out_mask, out_mask, out_mask], 1).cpu()
                out_mask = out_mask.cpu().numpy()
                real_B = real_B.cpu().numpy()
                real_B = real_B.astype(np.float32)
                for b in range(out_mask.shape[0]):
                    for h in range(out_mask.shape[2]):
                        for w in range(out_mask.shape[3]):
                            if out_mask[b,0,h,w] == 1:
                                real_B[b,:,h+real_B.shape[2]//2,w] = 0.3 * np.array([0.,0.,255.]) + 0.7 * real_B[b,:,h+real_B.shape[2]//2,w]
                            elif out_mask[b,0,h,w] == 2:
                                real_B[b,:,h+real_B.shape[2]//2,w] = 0.3 * np.array([0.,255.,0.]) + 0.7 * real_B[b,:,h+real_B.shape[2]//2,w]
                            elif out_mask[b,0,h,w] == 3:
                                real_B[b,:,h+real_B.shape[2]//2,w] = 0.3 * np.array([255.,0.,0.]) + 0.7 * real_B[b,:,h+real_B.shape[2]//2,w]

                real_B = torch.from_numpy(real_B).float().cuda()
                fake_image_list.append(real_B)
                fake_image_list.append(2 * (out_mask / (self.class_num - 1) - 0.5))
                fake_images = torch.cat(fake_image_list, dim=3)

                save_image(self.denorm(fake_images.data.cpu()),
                           os.path.join(self.result_path, '{}_fake.png'.format(i + 1)), nrow=1, padding=0)
                print('Translated images and saved into {}..!'.format(self.sample_path))

        print((time.time() - start) / len(self.data_loader))

    def evaluate(self):
        """Facial attribute transfer on CelebA or facial expression synthesis on RaFD."""
        # Load trained parameters
        G_path = os.path.join(self.model_save_path, '{}_G.pth'.format(self.test_model))

        self.G.load_state_dict(torch.load(G_path))
        self.G.eval()

        start = time.time()

        gt_num = 0
        img_num = 0
        overlap_num = 0
        
        with torch.no_grad():
            for i, (real_A,real_B) in enumerate(self.data_loader):
                print('i=',i)
                real_A = self.to_var(real_A)
                real_B = self.to_var(real_B)
                fake_B = self.G(real_A)
                upsample = nn.Upsample(size=(real_B.shape[1],real_B.shape[2]),mode='nearest')
                fake_B_up = upsample(fake_B)
                _, out_mask = torch.max(fake_B_up, dim=1)

                out_mask = out_mask.cpu().numpy()
                real_B = real_B.cpu().numpy()

                sample_pointnum = 40
                dis_thre = 0.03
                for n in range(out_mask.shape[0]):
                    for h in range(0, out_mask.shape[1], out_mask.shape[1]//sample_pointnum):
                        img_w_list = [[] for _ in range(self.class_num-1)]
                        gt_w_list = [[] for _ in range(self.class_num-1)]
                        
                        for w in range(out_mask.shape[2]):
                            if out_mask[n,h,w] > 0:
                                img_w_list[out_mask[n,h,w]-1].append(w)  
                            if real_B[n,h,w] > 0:
                                gt_w_list[real_B[n,h,w]-1].append(w)

                        for i in range(self.class_num-1):
                            if len(img_w_list[i]) > 0 and len(gt_w_list[i]) > 0 \
                                and abs(img_w_list[i][len(img_w_list[i])//2] - \
                                gt_w_list[i][len(gt_w_list[i])//2]) < dis_thre * out_mask.shape[2]:
                                overlap_num += 1                                

                            if len(img_w_list[i]) > 0 and img_w_list[i][len(img_w_list[i])//2] > 0:
                                img_num += 1

                            if len(gt_w_list[i]) > 0 and gt_w_list[i][len(gt_w_list[i])//2] > 0:
                                gt_num += 1

                recall = overlap_num / gt_num
                precision = overlap_num / img_num
                print('overlap_num=',overlap_num)
                print('gt_num=',gt_num)
                print('img_num=',img_num)
                print('recall={}'.format(recall))
                print('precision={}'.format(precision))

