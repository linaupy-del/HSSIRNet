import os
import glob
import sys
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data
import random


'''
通过继承Data.Dataset，实现将一组Tensor数据对封装成Tensor数据集
至少要重载__init__，__len__和__getitem__方法
'''


class Dataset(Data.Dataset):
    def __init__(self, filesmov,filesfix,filesatlas):
        # 初始化
        self.files_mov = filesmov
        self.files_fix = filesfix
        self.files_atlas =filesatlas 
        
    def __len__(self):
        # 返回数据集的大小
        return len(self.files_mov)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        mov_img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files_mov[index]))[np.newaxis, ...]   # (1, 160, 192, 160)
        fix_img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files_fix[index]))[np.newaxis, ...]
        atlas_img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files_atlas[0]))[np.newaxis, ...]
        # (1, 160, 192, 160)

        base_name = os.path.basename(self.files_fix[index]).replace('.nii.gz', '')
        trainsegDir = '../data_P/Brats20_OASIS1/Brats20/train/mask/' + base_name
        img_seg = sitk.GetArrayFromImage(sitk.ReadImage(trainsegDir + '_seg.nii.gz'))[np.newaxis, ...]
        imgseg_name = os.path.basename(trainsegDir + '_seg.nii.gz')
        
        # 返回值自动转换为torch的tensor类型
        return fix_img_arr,img_seg,mov_img_arr,atlas_img_arr,self.files_fix[index],self.files_mov[index],self.files_atlas[0],imgseg_name





