# -*- coding: utf-8 -*-
# @Time    : 2022/2/12 22:25
# @Author  : naptmn
# @File    : featureprocess.py
# @Software: PyCharm
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os
import time
import numpy as np
from torch.utils.data import dataloader, Dataset
from PIL import Image
def extract_feature(model, img, use_gpu=True):
    '''

    '''
    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    features = torch.FloatTensor()
    use_gpu = use_gpu and torch.cuda.is_available()
    img = Image.fromarray(img)
    img = img_transform(img)
    img = img.unsqueeze(0)
    img = img.cuda() if use_gpu else img
    input_img = Variable(img.cuda())
    outputs = model(input_img)
    ff = outputs.data.cpu()
    # norm feature
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    features = torch.cat((features, ff), 0)
    return features

def save_features(features, name):
    # 传入的应该是一个numpy数组
    name = './features/'+name
    np.save(name, features)

def load_model(pretrained_model=None, use_gpu=True):
    """

    :param check_point: Pretrained model path.
    :return:
    """
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    add_block = []
    add_block += [nn.Linear(num_ftrs, 30)]  #number of training classes
    model.fc = nn.Sequential(*add_block)
    model.load_state_dict(torch.load(pretrained_model))

    # remove the final fc layer
    model.fc = nn.Sequential()
    # change to test modal
    model = model.eval()
    use_gpu = use_gpu and torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    return model
