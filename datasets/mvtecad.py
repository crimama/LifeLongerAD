from glob import glob 
import os 
import random 
import numpy as np 
import pandas as pd 

import cv2 
from PIL import Image 
from typing import Literal
import torch 
from torch.utils.data import Dataset

class MVTecAD(Dataset):
    '''
    Example 
        df = get_df(
            datadir       = datadir ,
            class_name    = class_name,
            anomaly_ratio = anomaly_ratio
        )
        trainset = MVTecAD(
            df           = df,
            train_mode   = 'train',
            transform    = train_augmentation,
            gt_transform = gt_augmentation,
            gt           = True 
        )
    '''
    def __init__(self, df: pd.DataFrame, class_name:str, train_mode:str, transform, gt_transform):
        '''
        train_mode = ['train','valid','test']
        '''
        self.df = df 
        self.class_name = class_name 
        
        
        # train / test split 
        self.img_dirs = self.df[self.df['train/test'] == train_mode][0].values # column 0 : img_dirs 
        self.labels = self.df[self.df['train/test'] == train_mode]['anomaly'].values 
        
        # ground truth 
        self.gt_transform = gt_transform 

        # Image 
        self.transform = transform         
        self.name = 'MVTecAD'        
        self.train_mode = train_mode                    
        
    def _get_ground_truth(self, img_dir, img):
        img_dir = img_dir.split('/')
        if img_dir[-2] !='good':
            img_dir[-3] = 'ground_truth'
            img_dir[-1] = img_dir[-1].split('.')[0] + '_mask.png'
            img_dir = '/'.join(img_dir)
            gt = Image.open(img_dir)
            gt = self.gt_transform(gt)
        else:
            gt = torch.zeros([1, *img.size()[1:]])
        return gt
        
    def __len__(self):
        return len(self.img_dirs)
    
    
    def __getitem__(self,idx):
        img_dir = self.img_dirs[idx]        
        
        img = Image.open(img_dir).convert('RGB')     
        img = self.transform(img)
        img = img.type(torch.float32)
        label = self.labels[idx]
        
        if self.train_mode == 'test': # Test
            gt = self._get_ground_truth(img_dir,img)
            gt = (gt > 0).float()            
            
            return img, label, gt
        
                
        else:
            return img,label 
        
        
    