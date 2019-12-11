import torch
import os
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import glob
import cv2
import torch.nn.functional as F

def listAllFileName(rootDir, lines):
    dir=os.listdir(rootDir)
    for filename in dir:
        pathname = os.path.join(rootDir, filename)
        if (os.path.isfile(pathname)):
            if pathname.endswith('.bmp') or pathname.endswith('.jpg') or pathname.endswith('.png') \
                    or pathname.endswith('.JPG'):
                lines.append(pathname)
        else:
            listAllFileName(pathname,lines)

class CelebDataset(Dataset):
    def __init__(self, image_src, image_tgt, transform, mode, width, height):
        self.image_src = image_src
        self.image_tgt = image_tgt

        self.transform = transform
        self.mode = mode

        random.seed(1234)
        self.src_filenames = []
        listAllFileName(self.image_src,self.src_filenames)

        random.shuffle(self.src_filenames)

        self.num_data = len(self.src_filenames)#+len(self.src_filenames2)
        self.resize=(width,height)

    def __getitem__(self, index):
        path_name_src = self.src_filenames[index % len(self.src_filenames)]
        path_src, name_tgt = os.path.split(path_name_src)

        path_name_tgt = os.path.join(self.image_tgt, name_tgt.split(".jpg")[0] + "_1"+".png")

        image_src = Image.open(path_name_src)
        image_tgt = Image.open(path_name_tgt)
        w, h = image_src.size
        left, top, right, bottom = 0, h//2, w, h
        region = (left,top,right,bottom)
        image_src = image_src.crop(region).resize(self.resize, Image.NEAREST)
        image_tgt = image_tgt.crop(region).resize(self.resize, Image.NEAREST)

        image_tgt=np.asarray(image_tgt,dtype=np.int64)

        return self.transform(image_src), torch.from_numpy(image_tgt)

    def __len__(self):
        return self.num_data

def get_loader(image_src, image_tgt, batch_size=4, mode='train', width=128, height=128):
    if mode == 'train':
        transform = transforms.Compose([
            #transforms.RandomRotation()
            #transforms.CenterCrop(crop_size),
            #transforms.Resize(image_size, interpolation=Image.ANTIALIAS),
            #transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            # transforms.CenterCrop(crop_size),
            # transforms.Resize(image_size, interpolation=Image.ANTIALIAS),
            # transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = CelebDataset(image_src, image_tgt, transform, mode, width, height)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader