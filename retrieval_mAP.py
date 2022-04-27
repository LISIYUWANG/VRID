# -*- coding: utf-8 -*-
# @Time   : 2022/3/28 13:43
# @Author : LWDZ
# @File   : retrieval.py
# @aim    : image retrieval
#--------------------------------------------------

# -*- coding: utf-8 -*-
import os
import sys
from sklearn.preprocessing import MinMaxScaler
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from get_features import *
import PIL.Image as img
from pathlib import Path
import glob
import re
import pickle

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize,Normalizer

def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def img_trans(infile_path,outfile_folder,similarity,flag=True):
    im = img.open(infile_path)
    name = infile_path.split('/')[-1]
    if flag == False:
        similarity=str(similarity).split('.')[1]
    else:
        similarity = str(similarity)
    outfile = str(outfile_folder)+'/'+str(name)+'_'+similarity
    im.save(outfile)
def EuclideanDistances(a,b):
    sq_a = a**2
    sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()
    return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))
def compute_mAP(index, good_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc
    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc
# Evaluate
def evaluate_cos(qf,ql,gf,gl):
    
    fnorm = torch.norm(gf, p=2, dim=1, keepdim=True)
    gf = gf.div(fnorm.expand_as(gf))
    fnorm = torch.norm(qf, p=2, dim=1, keepdim=True)
    qf = qf.div(fnorm.expand_as(qf))
    
    score = gf*qf
    score = score.sum(1)
    # predict index
    s, index = score.sort(dim=0, descending=True)
    s = s.cpu().data.numpy()
    # good index
    ql = np.int64(ql)
    gl = np.array(gl)
    
    good_index = np.argwhere(gl==ql)
   
    CMC_tmp = compute_mAP(index, good_index)
    return CMC_tmp
def evaluate_rank1_2(qf,ql,gf,gl):
    
    gf2=gf
    fnorm = torch.norm(gf, p=2, dim=1, keepdim=True)
    gf = gf.div(fnorm.expand_as(gf))
    fnorm = torch.norm(qf, p=2, dim=1, keepdim=True)
    qf = qf.div(fnorm.expand_as(qf))
    score = gf*qf
    score = score.sum(1)
    # predict index
    s, index = score.sort(dim=0, descending=True)

    #feature2
    qf2 = gf2[index[0],:]
    qf2 = qf2.reshape(1,qf2.shape[0])
    fnorm = torch.norm(qf2, p=2, dim=1, keepdim=True)
    qf2 = qf2.div(fnorm.expand_as(qf2))
    score2=gf*qf2
    score2=score2.sum(1)
    score2=s[index[0]]*score2
    #score = score+ score2
    score = score*score2
    s, index = score.sort(dim=0, descending=True)
    s = s.cpu().data.numpy()
    
    # good index
    #ql = int(ql)
    ql = np.int64(ql)
    gl = np.array(gl)
    
    good_index = np.argwhere(gl==ql)
   
    CMC_tmp = compute_mAP(index, good_index)
    return CMC_tmp
def evaluate_Euclidean(qf,ql,gf,gl):
    
    score = EuclideanDistances(gf,qf)
    score = -score
    # predict index
    s, index = score.sort(dim=0, descending=True)
    s = s.cpu().data.numpy()
    # good index
    ql = np.int64(ql)
    gl = np.array(gl)
    good_index = np.argwhere(gl==ql)
    CMC_tmp = compute_mAP(index, good_index)
    return CMC_tmp
def evaluate_cos_len(qf,ql,gf,gl):
    
    gf_fnorm = torch.norm(gf, p=2, dim=1, keepdim=True)
    qf_fnorm = torch.norm(qf, p=2, dim=1, keepdim=True)
    fm_vec=qf_fnorm*torch.ones(gf.shape[0],1)
    mx = torch.maximum(gf_fnorm,fm_vec)
    mn = torch.minimum(gf_fnorm,fm_vec)
    one = torch.ones(gf.shape[0],1)
    score2 = mn.div(mx)-1
    
    gf = gf.div(gf_fnorm.expand_as(gf))
    qf = qf.div(qf_fnorm.expand_as(qf))
    score = gf*qf
    score = score.sum(1)
    normalizer = Normalizer(norm='l2').fit(score.reshape(1,gf.shape[0]))  #按行操作
    score2 = normalizer.transform(score2.reshape(1,gf.shape[0]))
    
    score = score+score2[0]#.reshape(1,gf.shape[0])[0]
    # predict index
    s, index = score.sort(dim=0, descending=True)
    s = s.cpu().data.numpy()
    
    # good index
    ql = np.int64(ql)
    gl = np.array(gl)
    
    good_index = np.argwhere(gl==ql)
   
    CMC_tmp = compute_mAP(index, good_index)
    return CMC_tmp

def main_worker(args):

    print('data_name :',args.data_name)
    use_gpu = args.use_gpu
    model_path = args.model_path
    
    gallery_img_feature_path = args.feature_path+'/gallery_'+args.data_name+'_nodiv_features.pkl'
    gallery_color_feature_path =args.feature_path+'/gallery_'+args.color_data_name+'_nodiv_features.pkl'
    gallery_feature_label_path = args.feature_path+'/gallery_'+args.data_name+'_nodiv_labels.pkl'
    
    query_img_path = args.query_data_path +'/'+args.data_name+'/query'
    query_color_path = args.query_data_path +'/'+args.color_data_name+'/query'
    # Prepare data.

    # get gallery features.
    gallery_feature_label = pickle.load(open(gallery_feature_label_path, 'rb'))
    gallery_img_feature = pickle.load(open(gallery_img_feature_path, 'rb'))

    if args.if_concat:
        gallery_color_feature = pickle.load(open(gallery_color_feature_path, 'rb'))
        gallery_img_feature = torch.cat((gallery_img_feature,gallery_color_feature),1)

#     # Query.
#     # Extract query features.
    CMC = torch.IntTensor(len(gallery_feature_label)).zero_()
    ap = 0.0
    k = 0
    time_all = 0
    for img_name in os.listdir(query_img_path):
        k += 1
        # begin to query
        time_begin = time.time()
        img_path = os.path.join(query_img_path,img_name)
        color_img_path = os.path.join(query_color_path,img_name)
        if not os.path.exists(color_img_path):
            color_img_path = os.path.join(query_color_path,img_name.split('.')[0]+'.png')
        query_feature = extract_feature_query_nodiv(img_path, model_path)
        if args.if_concat:
            query_color_feature = extract_feature_query_nodiv(color_img_path, model_path)
            query_feature = torch.cat((query_feature,query_color_feature),1)

        label = img_name.split('_')[0]
        gallery_feature_label = list(map(int, gallery_feature_label)) 
        ap_tmp, CMC_tmp = evaluate_cos(query_feature, label,gallery_img_feature,gallery_feature_label)
        # end query
        time_query = time.time()
        time_all = time_all + (time_query-time_begin)
        #break
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
  
    average_time = time_all*1.0 / len(os.listdir(query_img_path))
    CMC = CMC.float()
    CMC = CMC / len(os.listdir(query_img_path))  # average CMC
    result_df = pd.DataFrame({})
    result_df['Rank1']= float(CMC[0])
    result_df['Rank5']= float(CMC[4])
    result_df['Rank10']= float(CMC[9])
    result_df['mAP']= ap / len(os.listdir(query_img_path))
    result_df['average_time']=average_time
    result_df.to_csv('results/query_result/'+args.data_name +'.csv',index=False)
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f  average_time:%f' % (CMC[0], CMC[4], CMC[9], ap / len(os.listdir(query_img_path)) ,average_time) )
    
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="retrieval images")
    # data
    parser.add_argument('--model_path',  default='./models/net_best.pth', help="path of feature extracting models")
    parser.add_argument('--feature_path',default='results/features/', help="gallery feature path")
    parser.add_argument('--data_name',default='', help="gallery  folder")
    parser.add_argument('--color_data_name',default='', help="gallery color folder")
    
    # parser.add_argument('--gallery_img_name', default='', help="gallery image folder")
    parser.add_argument('--query_data_path', default='dataset/', help="query images path")
    # parser.add_argument('--query_img_name', default='', help="query image folder name")
    parser.add_argument('--if_concat', default=False, type = bool,help="query image folder name")
    parser.add_argument('--use_gpu', default=True, help="if use gpu ")
    args = parser.parse_args()

    main_worker(args)


