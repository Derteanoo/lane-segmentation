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
    def __init__(self, image_src, transform, mode, width, height):
        self.image_src = image_src

        self.transform = transform
        self.mode = mode

        self.src_filenames = []
        listAllFileName(self.image_src,self.src_filenames)
        self.src_filenames.sort()

        self.num_data = len(self.src_filenames)

        self.resize=(width, height)

    def __getitem__(self, index):

        path_name_src = self.src_filenames[index % len(self.src_filenames)]
        
        image_src = Image.open(path_name_src)#.resize(self.resize, Image.BILINEAR)
        w, h = image_src.size
        left, top, right, bottom = 0, h//2, w, h
        region = (left,top,right,bottom)
        image_half = image_src.crop(region).resize(self.resize, Image.BILINEAR)

        return self.transform(image_half), self.transform(image_src)

    def __len__(self):
        return self.num_data


def get_loader(image_src, batch_size=1, mode='test', width=128, height=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = CelebDataset(image_src, transform, mode, width, height)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers = 32)
    return data_loader