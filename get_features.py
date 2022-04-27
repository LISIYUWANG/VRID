# -*- coding: utf-8 -*-
# @Time   : 2022/3/28 13:46
# @Author : LWDZ
# @File   : get_features.py
# @aim    : save and get features
#--------------------------------------------------

import numpy as np
import os
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch.utils.data import dataloader, Dataset
import argparse
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from PIL import Image

class Gallery(Dataset):
    """
    Images in database.
    """

    def __init__(self, image_paths, transform=None):
        super().__init__()

        self.image_paths = image_paths
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, image_path

    def __len__(self):
        return len(self.image_paths)

def get_file_list(file_path_list, sort=True):
    """
    Get list of file paths in one folder.
    :param file_path: A folder path or path list.
    :return: file list: File path list of
    """
    import random
    if isinstance(file_path_list, str):
        file_path_list = [file_path_list]
    file_lists = []
    for file_path in file_path_list:
        assert os.path.isdir(file_path)
        file_list = os.listdir(file_path)
        if sort:
            file_list.sort()
        else:
            random.shuffle(file_list)
        file_list = [file_path + file for file in file_list]
        file_lists.append(file_list)
    if len(file_lists) == 1:
        file_lists = file_lists[0]
    return file_lists

def load_data(data_path, batch_size=1, shuffle=False, transform='default'):
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]) if transform == 'default' else transform

    image_path_list = get_file_list(data_path)
    gallery_data = Gallery(image_paths=image_path_list,
                           transform=data_transform,
                           )

    data_loader = dataloader.DataLoader(dataset=gallery_data,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_workers=0,
                                        )
    return data_loader

def load_model(pretrained_model_path=None, use_gpu=True):
    """
    :param check_point: Pretrained models path.
    :return:
    """
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    add_block = []
    add_block += [nn.Linear(num_ftrs, 30)]  #number of training classes
    model.fc = nn.Sequential(*add_block)
    model.load_state_dict(torch.load(pretrained_model_path))
    
    # remove the final fc layer
    model.fc = nn.Sequential()  #output size : 2048
    # change to test modal
    model = model.eval()
    use_gpu = use_gpu and torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    return model

def load_query_image(query_path):
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    query_image = datasets.folder.default_loader(query_path)
    query_image = data_transforms(query_image)
    return query_image

def extract_feature_query(img_path,model_path, use_gpu=True):
    model = load_model(pretrained_model_path=model_path, use_gpu=True)
    model = nn.DataParallel(model)
    # Query.
    query_image = load_query_image(img_path)
    c, h, w = query_image.size()
    img = query_image.view(-1,c,h,w)
    use_gpu = use_gpu and torch.cuda.is_available()
    img = img.cuda() if use_gpu else img
    input_img = Variable(img)
    outputs = model(input_img)
    ff = outputs.data.cpu()
    fnorm = torch.norm(ff,p=2,dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    return ff
def extract_feature_query_nodiv(img_path,model_path, use_gpu=True):
    model = load_model(pretrained_model_path=model_path, use_gpu=True)
    model = nn.DataParallel(model)
    # Query.
    query_image = load_query_image(img_path)
    c, h, w = query_image.size()
    img = query_image.view(-1,c,h,w)
    use_gpu = use_gpu and torch.cuda.is_available()
    img = img.cuda() if use_gpu else img
    input_img = Variable(img)
    outputs = model(input_img)
    ff = outputs.data.cpu()
    return ff
def extract_features(model_path='./models/net_best.pth',data_path='./dataset/data',data_name = '',bach_size=256,use_gpu=True):
    #data_folder_name = data_path.split('/')[-1]
    data_path = os.path.join(data_path,data_name)
    print('data_path :',data_path)
    data_loader = load_data(
                            data_path=data_path+'/', #用于特征比对的处理后图片
                            batch_size=bach_size,
                            shuffle=False,
                            transform='default',
                            )
    # Prepare models.
    model = load_model(pretrained_model_path = model_path, use_gpu=True)
    model = nn.DataParallel(model)
    # Extract database features.
    features = torch.FloatTensor()
    label_list = []
    use_gpu = use_gpu and torch.cuda.is_available()
    for img, path in data_loader:
        img = img.cuda() if use_gpu else img
        input_img = Variable(img.cuda())
        outputs = model(input_img)
        ff = outputs.data.cpu()
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
        path = list(path)
        label = [(tp.split('/')[-1]).split('_')[0] for tp in path]
        label_list += list(label)
    return features,label_list
def extract_features_nodiv(model_path='./models/net_best.pth',data_path='./dataset/data',data_name = '',bach_size=256,use_gpu=True):
    #data_folder_name = data_path.split('/')[-1]
    data_path = os.path.join(data_path,data_name)
    print('data_path :',data_path)
    data_loader = load_data(
                            data_path=data_path+'/', #用于特征比对的处理后图片
                            batch_size=bach_size,
                            shuffle=False,
                            transform='default',
                            )
    # Prepare models.
    model = load_model(pretrained_model_path = model_path, use_gpu=True)
    model = nn.DataParallel(model)
    # Extract database features.
    features = torch.FloatTensor()
    label_list = []
    use_gpu = use_gpu and torch.cuda.is_available()
    for img, path in data_loader:
        img = img.cuda() if use_gpu else img
        input_img = Variable(img.cuda())
        outputs = model(input_img)
        ff = outputs.data.cpu()
        # norm feature
        #fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        #ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
        path = list(path)
        label = [(tp.split('/')[-1]).split('_')[0] for tp in path]
        label_list += list(label)
    return features,label_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare the gallery feature")
    # data
    parser.add_argument('-b', '--batch_size', type=int, default=100)
    parser.add_argument('--model_path',  default='./models/net_best.pth', help="path of feature extracting models")
    parser.add_argument('--data_path',default='dataset', help="data path")
    parser.add_argument('--data_name', default='gallery', help="data name")
    parser.add_argument('--use_gpu', default=True, help="if use gpu ")
    args = parser.parse_args()
    # gallery feature
    data = 'div_color_cut_seg_all_person'
    features,labels = extract_features_nodiv(args.model_path,args.data_path+'/'+data,args.data_name,args.batch_size,args.use_gpu)
    pickle.dump(features, open('results/features/gallery_'+data+'_nodiv_features.pkl', 'wb'))
    pickle.dump(labels, open('results/features/gallery_'+data+'_nodiv_labels.pkl', 'wb'))


    