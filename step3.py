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
from Networks import inpainting_reg_syn_net
# python imports
from re import A
import warnings
import sys
# external imports
from torch.optim import Adam
import torch.utils.data as Data
import torch.nn.functional as F
from generators4 import Dataset
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
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.test_result, name))

    
def train(args):
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    log_name = 'zuhui'
    print("log_name: ", log_name)
    f = open(os.path.join(args.Log_dir, log_name + ".txt"), "a")  
    
    model = inpainting_reg_syn_net.Net(for_train=True,use_checkpoint=True)
    model.to(device)
    model.load_state_dict(torch.load('model/pre_reg_syn/358.pth',map_location='cuda:0'))
    model.train() 
    opt = Adam(model.parameters(), lr=args.lr)
    
    train_files_mov = glob.glob(os.path.join(args.train_dir_mov,'image', '*.nii'))
    train_files_fix = glob.glob(os.path.join(args.train_dir_fix, 'image', '*.nii.gz'))
    train_files_atlas = glob.glob(os.path.join(args.train_dir_atlas, '*.nii.gz'))
    
    DS = Dataset(filesmov = train_files_mov,filesfix=train_files_fix,filesatlas=train_files_atlas) 
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
   
    for i in range (1,args.epoch + 1 ):
        for data in DL:
            input_fixed,input_mask,input_moving,input_atlas,fix_name,mov_name,atlas_name,seg_name = data
            
            input_fixed = input_fixed.to(device).float()
            input_mask = input_mask.to(device).float()
            input_moving = input_moving.to(device).float()
            input_atlas =  input_atlas.to(device).float()

            mask = input_mask
            mask_transformed = torch.where(torch.isnan(mask), torch.tensor(0.0, device=mask.device), mask)
            S = torch.where(mask_transformed != 0, torch.tensor(1.0, device=input_mask.device), mask_transformed)#肿瘤区域
            A = 1 - S #正常区域
            
            warped,sysimg,flow,FP,warpedP,FPA,FPS = model(input_moving,input_fixed,input_atlas,input_mask,A,S)

            DICELOSS = losses.Dice().loss
            dice = DICELOSS(warped,FP)
            
            
            #配准loss
            sim_loss =losses.ncc_loss(warped,FP)
            grad_loss = losses.gradient_loss(flow) 
        
            Mse_loss = losses.mse_loss
            sys_loss =  Mse_loss(warped,sysimg)
            
            #总loss
            loss = sim_loss + grad_loss+sys_loss+ dice
            # loss = sim_loss + grad_loss
            print("i: %d fix: %s mov: %s atlas: %s seg: %s loss: %f   sim: %f  L2: %f  sys: %f"
                  % (i, fix_name,mov_name,atlas_name,seg_name,loss.item(),sim_loss.item(), grad_loss.item(),sys_loss.item()),flush=True)                                                                                                         
            print("%d, %s, %s, %s, %s, %f, %f, %f, %f"
                  % (i, fix_name,mov_name,atlas_name,seg_name,loss.item(),sim_loss.item(), grad_loss.item(),sys_loss.item()),file=f)
            
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
                    dest="gpu", default='0')#在hxy屏幕上跑的
    parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=1e-4)
    parser.add_argument("--epoch", type=int, help="number of iterations",
                    dest="epoch", default=10000)
    parser.add_argument("--batch_size", type=int, help="batch_size",
                    dest="batch_size", default=1)
    parser.add_argument("--alpha", type=float, help="regularization parameter",
                    dest="alpha", default=1.0)
    parser.add_argument("--train_dir_mov", type=str, help="data folder with training vols",
                    dest="train_dir_mov", default="/mnt/e3ef8c10-8778-416f-b630-481836b748c9/hxy/code/data_P/Brats20_OASIS1/OASIS1/train/")
    parser.add_argument("--train_dir_fix", type=str, help="data folder with training vols",
                    dest="train_dir_fix", default="/mnt/e3ef8c10-8778-416f-b630-481836b748c9/hxy/code/data_P/Brats20_OASIS1/Brats20/train/")

    parser.add_argument("--train_dir_atlas", type=str, help="data folder with training vols",
                    dest="train_dir_atlas", default="/mnt/e3ef8c10-8778-416f-b630-481836b748c9/hxy/code/data_P/Brats20_OASIS1/muban/")

    parser.add_argument("--model_dir", type=str, help="data folder with training vols",
                    dest="model_dir", default="model2/step3")
    parser.add_argument("--Log_dir", type=str, help="data folder with training vols",
                    dest="Log_dir", default="logs2/step3")
    args = parser.parse_args()
    train(args)