# -*- coding: utf-8 -*-
# @Time   : 2022/3/28 15:55
# @Author : LWDZ
# @File   : caculate_color_num.py
# @aim    : get color number image
#--------------------------------------------------
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
import cv2
import os
import numpy as np

def get_color_num_fig(from_path,save_path,img_name):
    img_path = os.path.join(from_path,img_name)
    img = cv2.imread(img_path)
    img = np.array(img)
    ls = np.zeros((54, 54, 54), dtype=int)
    h = img.shape[0]
    w = img.shape[1]
    mx = 0
    for i in range(h):
        for j in range(w):
            if ls[int(img[i, j, 0] / 5), int(img[i, j, 1] / 5), int(img[i, j, 2] / 5)] > 400:
                continue
            ls[int(img[i, j, 0] / 5), int(img[i, j, 1] / 5), int(img[i, j, 2] / 5)] += 1
            if mx < ls[int(img[i, j, 0] / 5), int(img[i, j, 1] / 5), int(img[i, j, 2] / 5)] and \
                    int(img[i, j, 0] / 5) + int(img[i, j, 1] / 5) + int(img[i, j, 2] / 5) > 0 and \
                    int(img[i, j, 0] / 5) + int(img[i, j, 1] / 5) + int(img[i, j, 2] / 5) < 153:
                mx = ls[int(img[i, j, 0] / 5), int(img[i, j, 1] / 5), int(img[i, j, 2] / 5)]
    if mx < 50:
        return
    mxar = 255 * 1.0 / mx
    ls[0, 0, 0] = 0
    ls = ls.reshape(243, 216, 3)
    ls = ls * mxar
    cv2.imwrite(save_path+'/'+img_name, ls)


def change_dir(name_from_path,img_from_path,save_path):
    for i in os.listdir(name_from_path):
        img_save = os.path.join(save_path,i)
        img_from = os.path.join(img_from_path,i)
        img = cv2.imread(img_from)
        cv2.imwrite(img_save,img)

if __name__ == '__main__':
    data_name = 'seg_all_person'
    data_dir='../dataset/'+data_name
    color_dir='../dataset/color_'+data_name

    if not os.path.exists(color_dir):
        os.mkdir(color_dir)
        img_paths = os.path.join(data_dir,i)
        save_img_paths = os.path.join(color_dir, i)
        if not os.path.exists(save_img_paths):
            os.mkdir(save_img_paths)
        for j in os.listdir(img_paths):
            img_path = os.path.join(img_paths, j)
            get_color_num_fig(img_paths,save_img_paths,j)

