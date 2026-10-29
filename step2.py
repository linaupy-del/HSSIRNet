
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
import re
from Networks import pre_reg_syn_net_ex2

from re import A
import warnings
import sys

from torch.optim import Adam
import torch.utils.data as Data

import torch.nn.functional as F
import losses
from generators_ex2_1 import Dataset

import torchvision



def make_dirs():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.Log_dir):
        os.makedirs(args.Log_dir)
    # if not os.path.exists(args.result):
    #     os.makedirs(args.result)
            
def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    # img = sitk.GetImageFromArray(img)
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.test_result, name))

def train(args):
    make_dirs()
    # writer = SummaryWriter(args.Log_dir)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    log_name = 'DICE_step2'
    print("log_name: ", log_name)
    f = open(os.path.join(args.Log_dir, log_name + ".txt"), "a")  
    
    model = pre_reg_syn_net_ex2.Net(for_train=True,use_checkpoint=True)
    model.to(device)

    STN_label = pre_reg_syn_net_ex2.SpatialTransformer_block(mode='nearest') 
    STN_label.to(device)

    model.load_state_dict(torch.load('model2/step1/DICE/186.pth',map_location='cuda:0'))

    model.train()
    opt = Adam(model.parameters(), lr=args.lr)
   
    
    train_files_mov = glob.glob(os.path.join(args.train_dir_mov,'train','image','*.nii'))
    train_files_fix = glob.glob(os.path.join(args.train_dir_fix,'train', 'image', '*.nii.gz'))
    
    DS = Dataset(filesmov = train_files_mov,filesfix=train_files_fix) 
    print("Number of training images: ", len(DS))
    DL = Data.DataLoader(DS, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    for i in range (1,args.epoch + 1 ):
        for data in DL:
            
            # input_fixed,input_mask,input_moving,fix_name,mov_name,mask_name = data
            input_fixed,input_mask,input_fix_label,input_moving,input_mov_label,fix_name,fixlabel_name,movlabel_name,mov_name,mask_name = data
            
            input_moving = input_moving.to(device).float()
            input_fixed = input_fixed.to(device).float()
            input_mask = input_mask.to(device).float()
            input_fix_label =input_fix_label.to(device).float()
            input_mov_label = input_mov_label.to(device).float()


            
            flow,warped,sysimg= model(input_moving,input_fixed)

            warped_label = STN_label(input_mov_label,flow)
            
            
            
            mask = input_mask
            mask_transformed = torch.where(torch.isnan(mask), torch.tensor(0.0, device=mask.device), mask)
            S = torch.where(mask_transformed != 0, torch.tensor(1.0, device=input_mask.device), mask_transformed)
            A = 1 - S
            
            sim_loss =losses.ncc_loss(warped*A,input_fixed*A)
            grad_loss = losses.gradient_loss(flow) 
            
            Mse_loss = losses.mse_loss
            sys_loss_normal =  Mse_loss(sysimg*A,warped*A)

            DICELOSS = losses.Dice().loss
            dice = DICELOSS(warped_label*A,input_fix_label*A)
            
            #总loss
            loss = sim_loss + grad_loss  + sys_loss_normal + dice
           
            
            print("i: %d fix: %s mov: %s mask: %s loss: %f grad_loss:%f sim_loss:%f sys_loss_normal:%f dice:%f"
                  % (i, fix_name,mov_name,mask_name, loss.item(), grad_loss.item(),sim_loss.item(),sys_loss_normal.item(),dice.item()),flush=True)                                                                                                         
            print("%d, %s, %s, %s %f, %f, %f, %f,%f"
                  % (i, fix_name,mov_name,mask_name, loss.item(), grad_loss.item(),sim_loss.item(),sys_loss_normal.item(),dice.item()),file=f)
            
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
                    dest="model_dir", default="model2/step2/2")
    parser.add_argument("--Log_dir", type=str, help="data folder with training vols",
                    dest="Log_dir", default="logs2/step2/2")

    args = parser.parse_args()
    train(args)

