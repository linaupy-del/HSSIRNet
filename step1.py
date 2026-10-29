import os
import glob
# external imports
import torch
import numpy as np
import time
import SimpleITK as sitk
# internal imports
from write_excel import *
import argparse
import losses
import re
from Networks import only_reg_net_exe2

# python imports
import os
import glob
from re import A
import warnings
import sys
# external imports

from torch.optim import Adam
import torch.utils.data as Data
import torch.nn.functional as F
# internal imports
# import tensorboard
# from torch.utils.tensorboard import SummaryWriter
from generators_ex2_1 import Dataset

import torchvision

def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.Log_dir):
        os.makedirs(args.Log_dir)
    # if not os.path.exists(args.test_result):
    #     os.makedirs(args.test_result)
            
def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    # img = sitk.GetImageFromArray(img)
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.test_result, name))
    # save_path = '/mnt/e3ef8c10-8778-416f-b630-481836b748c9/hxy/code/new/mask/2'
    # sitk.WriteImage(img, os.path.join(save_path, name))

# def Dice(vol1, vol2, labels=None, nargout=1):
    
#     if labels is None:
#         labels = np.unique(np.concatenate((vol1, vol2)))
#         labels = np.delete(labels, np.where(labels == 0))  # remove background

#     if len(labels)==0:
#         return 0.0
        
#     dicem = np.zeros(len(labels))
    
#     for idx, lab in enumerate(labels):
#         vol1l = vol1 == lab
#         vol2l = vol2 == lab
#         top = 2 * np.sum(np.logical_and(vol1l, vol2l))
#         bottom = np.sum(vol1l) + np.sum(vol2l)
#         bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
#         dicem[idx] = top / bottom

#     if nargout == 1:
#         # return np.mean(dicem)
#         return dicem
#     else:
#         return (dicem, labels)
import torch

def Dice(vol1, vol2, labels=None, nargout=1):
    # 确保输入是 PyTorch 张量，如果不是则进行转换
    if isinstance(vol1, np.ndarray):
        vol1 = torch.tensor(vol1, device='cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(vol2, np.ndarray):
        vol2 = torch.tensor(vol2, device='cuda' if torch.cuda.is_available() else 'cpu')

    if labels is None:
        labels = torch.unique(torch.cat((vol1.flatten(), vol2.flatten())))
        labels = labels[labels != 0]  # remove background

    if len(labels) == 0:
        return torch.tensor(0.0, device=vol1.device)  # Ensure the return tensor is on the same device

    dicem = torch.zeros(len(labels), device=vol1.device)
    
    for idx, lab in enumerate(labels):
        vol1l = (vol1 == lab)
        vol2l = (vol2 == lab)
        top = 2 * torch.sum(vol1l & vol2l)
        bottom = torch.sum(vol1l) + torch.sum(vol2l)
        bottom = torch.max(bottom, torch.finfo(torch.float32).eps)  # add epsilon
        dicem[idx] = top / bottom

    if nargout == 1:
        return torch.mean(dicem)
    else:
        return dicem, labels

def train(args):
    make_dirs()
    # writer = SummaryWriter(args.Log_dir)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    log_name = 'ex2_onlyreg222'
    print("log_name: ", log_name)
    f = open(os.path.join(args.Log_dir, log_name + ".txt"), "a")  
     
    model = only_reg_net_exe2.Net(for_train=True,use_checkpoint=True)
    model.to(device)
    model.load_state_dict(torch.load('model2/step1/DICE3/20.pth'))########加入配准的模型代码
    model.train()
    opt = Adam(model.parameters(), lr=args.lr)
   
    
    train_files_mov = glob.glob(os.path.join(args.train_dir_mov,'train','image','*.nii'))
    train_files_fix = glob.glob(os.path.join(args.train_dir_fix,'train', 'image', '*.nii.gz'))
    
    DS = Dataset(filesmov = train_files_mov,filesfix=train_files_fix) 
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    for i in range (21,args.epoch + 1 ):
        for data in DL:

            input_fixed,input_mask,input_fix_label,input_moving,input_mov_label,fix_name,fixlabel_name,movlabel_name,mov_name,mask_name = data
            
            input_moving = input_moving.to(device).float()
            input_fixed = input_fixed.to(device).float()
            input_mask = input_mask.to(device).float()

            input_fix_label = input_fix_label.to(device).float()
            input_mov_label = input_mov_label.to(device).float()

            flow,warped,warp_label = model(input_moving,input_fixed,input_fix_label,input_mov_label)


            mask = input_mask
            mask_transformed = torch.where(torch.isnan(mask), torch.tensor(0.0, device=mask.device), mask)
            S = torch.where(mask_transformed != 0, torch.tensor(1.0, device=input_mask.device), mask_transformed)
            A = 1 - S
            
            sim_loss =losses.ncc_loss(warped*A,input_fixed*A)
            grad_loss = losses.gradient_loss(flow*A) 
            
            DICELOSS = losses.Dice().loss

            dice = DICELOSS(warp_label*A,input_fix_label*A)

            flow1 = flow*A
            NJ_loss = losses.NJ_loss()(flow1.permute(0, 2, 3, 4, 1))

            loss = grad_loss + 0.00001 * NJ_loss + sim_loss + dice

            
            print("i: %d fix: %s mov: %s mask: %s loss: %f grad_loss:%f NJ_loss:%f sim_loss:%f dice:%f"
                  % (i, fix_name,mov_name,mask_name, loss.item(), grad_loss.item(),NJ_loss.item(),sim_loss.item(),dice.item()),flush=True)                                                                                                         
            print("%d, %s, %s, %s %f, %f, %f, %f, %f"
                  % (i, fix_name,mov_name,mask_name, loss.item(), grad_loss.item(),NJ_loss.item(),sim_loss.item(),dice.item()),file=f)          

            opt.zero_grad() 
            loss.backward() 
            opt.step()
            
        if (i % 2 == 0):
            save_file_name = os.path.join(args.model_dir, '%d.pth' % i)
            torch.save(model.state_dict(), save_file_name)
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, help="gpu id",
                    dest="gpu", default='0')#这里从1修改到0
    parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=1e-4)
    parser.add_argument("--epoch", type=int, help="number of iterations",
                    dest="epoch", default=10000)
    parser.add_argument("--batch_size", type=int, help="batch_size",
                    dest="batch_size", default=1)
    parser.add_argument("--alpha", type=float, help="regularization parameter",
                    dest="alpha", default=1.0)
    parser.add_argument("--train_dir_mov", type=str, help="data folder with training vols",
                    dest="train_dir_mov", default="../data_P/Brats20_OASIS1/OASIS1/")
    parser.add_argument("--train_dir_fix", type=str, help="data folder with training vols",
                    dest="train_dir_fix", default="../data_P/Brats20_OASIS1/Brats20/")
    parser.add_argument("--model_dir", type=str, help="data folder with training vols",
                    dest="model_dir", default="/mnt/e3ef8c10-8778-416f-b630-481836b748c9/hxy/code/new3/model2/step1/DICE3")
    parser.add_argument("--Log_dir", type=str, help="data folder with training vols",
                    dest="Log_dir", default="/mnt/e3ef8c10-8778-416f-b630-481836b748c9/hxy/code/new3/logs2/step1/DICE3")
    args = parser.parse_args()
    train(args)

