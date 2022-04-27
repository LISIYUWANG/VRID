# -*- coding: utf-8 -*-
# @Time   : 2022/3/28 15:55
# @Author : LWDZ
# @File   : cut_person.py
# @aim    : cut image to get upper body
#--------------------------------------------------
import os
import cv2
import numpy as np

def cut_fig(from_path,save_path):

    for i in os.listdir(from_path):
        img_path=os.path.join(from_path,i)
        img = cv2.imread(img_path)
        img = np.array(img)
        h=img.shape[0]
        w = img.shape[1]
        trip_img = img[int(w / 8):int(5 * w /4), :]
        img_save_path = os.path.join(save_path,i)
        cv2.imwrite(img_save_path,trip_img)


def img_change_type(from_path,save_path):

    for i in os.listdir(from_path):
        k = i.split('.')[0]+'.png'
        img_path=os.path.join(from_path,i)
        print(img_path)
        img = cv2.imread(img_path)
        img_save_path = os.path.join(save_path,k)
        cv2.imwrite(img_save_path,img)

if __name__ == '__main__':

    data_name = 'all_person'
    from_path = '../dataset/'+data_name
    save_path = '../dataset/cut_'+data_name
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for j in os.listdir(from_path):
        folder = os.path.join(from_path,j)
        save_folder = os.path.join(save_path,j)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        cut_fig(folder,save_folder)
