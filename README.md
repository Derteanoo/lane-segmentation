# lane-segmentation

## Introduction

This repository contains some models for lane semantic segmentation and the pipeline of training and testing models, implemented in PyTorch.

![picc](https://github.com/Derteanoo/lane-segmentation/blob/master/stargan/results/28_fake.png?raw=true)

## Datasets
We train and test the model on our own dataset in the same format as the Tusimple dataset. You need to generate instance segmentation labels first.

## Test
We provide trained model, and it is saved in "savefile" directory. You can test as following:

`sh test.sh`

## Train
You can train the model as following:

`sh train.sh`
