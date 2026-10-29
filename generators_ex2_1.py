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
    def __init__(self, filesmov,filesfix):
        # 初始化
        self.files_mov = filesmov
        self.files_fix = filesfix
        
    def __len__(self):
        # 返回数据集的大小
        return len(self.files_mov)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        mov_img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files_mov[index]))[np.newaxis, ...]   # (1, 160, 192, 160)
        fix_img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files_fix[index]))[np.newaxis, ...]

        base_name = os.path.basename(self.files_fix[index]).replace('.nii.gz', '')

        trainsegDir = '/mnt/e3ef8c10-8778-416f-b630-481836b748c9/hxy/code/data_P/Brats20_OASIS1/Brats20/train/mask/' + base_name
        img_seg = sitk.GetArrayFromImage(sitk.ReadImage(trainsegDir + '_seg.nii.gz'))[np.newaxis, ...]
        imgseg_name = os.path.basename(trainsegDir + '_seg.nii.gz')

        fix_label_Dir = '/mnt/e3ef8c10-8778-416f-b630-481836b748c9/hxy/code/data_P/Brats20_OASIS1/Brats20/train/label/' + base_name
        fix_label= sitk.GetArrayFromImage(sitk.ReadImage(fix_label_Dir + '_temp_seg_seg.nii.gz'))[np.newaxis, ...]
        fixlabel_name = os.path.basename(fix_label_Dir + '_temp_seg_seg.nii.gz')

        base_name1 = os.path.basename(self.files_mov[index]).replace('.nii', '')
        mov_label_Dir = '/mnt/e3ef8c10-8778-416f-b630-481836b748c9/hxy/code/data_P/Brats20_OASIS1/OASIS1/train/label/'+ base_name1
        mov_label= sitk.GetArrayFromImage(sitk.ReadImage(mov_label_Dir + '.nii_label_seg.nii.gz'))[np.newaxis, ...]
        movlabel_name = os.path.basename(mov_label_Dir + '.nii_label_seg.nii.gz')      
        
        # 返回值自动转换为torch的tensor类型
        return fix_img_arr,img_seg,fix_label,mov_img_arr,mov_label,self.files_fix[index],fixlabel_name,movlabel_name,self.files_mov[index],imgseg_name

