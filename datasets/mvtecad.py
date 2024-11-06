from glob import glob 
import os 
import random 
import numpy as np 
import pandas as pd 

import cv2 
from PIL import Image 

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
    def __init__(self, df: pd.DataFrame, class_name:str, caption_dict:dict, train_mode:str, transform, gt_transform, 
                 num_neg_sample=1, gt=True, idx=False, text=True):
        '''
        train_mode = ['train','valid','test']
        '''
        self.df = df 
        self.class_name = class_name 
        
        # train / test split 
        self.img_dirs = self.df[self.df['train/test'] == train_mode][0].values # column 0 : img_dirs 
        self.labels = self.df[self.df['train/test'] == train_mode]['anomaly'].values 
        
        # ground truth 
        self.gt = gt # mode 
        self.gt_transform = gt_transform 

        # Image 
        self.transform = transform         
        self.name = 'MVTecAD'        
        self.idx = idx 
        self.train_mode = train_mode
        
        # Text 
        self.text_format = ["a photo of {}", "a picture of {}", "a image of {}"]
        self.positive, self.negative = self.caption_split(caption_dict, class_name)
        self.num_neg_sample = num_neg_sample
        np.random.shuffle(self.negative)
    def caption_split(self, caption_dict, class_name):
        '''
            Output 
                positive : dict 
                negative : list 
        '''
        task_order = list(caption_dict.keys()).index(class_name)
        self.task_order = task_order 
        negative = [] 
        self.negative_class = [] 
        for cn in list(caption_dict.keys())[:task_order]:
            caption = caption_dict[cn].values()
            negative.extend(caption)
            self.negative_class.append(cn)
        positive = caption_dict[class_name]
        return positive, negative 
        
        
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
    
    def get_easy_text(self, class_name):
        txt_formula = ['A photo of {}', 'A picture of {}', 'A image of {}']
        text = random.sample(txt_formula,1)[0].format(class_name)
        return text
    
    def __getitem__(self,idx):
        img_dir = self.img_dirs[idx]        
        
        img = Image.open(img_dir).convert('RGB')     
        img = self.transform(img)
        img = img.type(torch.float32)
        label = self.labels[idx]
        
        if self.gt: # Test
            gt = self._get_ground_truth(img_dir,img)
            gt = (gt > 0).float()            
            
            return img, label, gt
        
        else: # Train 
            if self.task_order == 0:
                negative_text = np.random.choice(self.text_format).format(self.class_name)
                positive_text = "a photo of {}".format(self.class_name)
            else:
                data_id = img_dir.split('/')[-1].strip('.png')
                positive_text = self.positive[data_id]
                negative_text = random.sample(self.negative,self.num_neg_sample) 
                
                # negative_text = self.get_easy_text(random.sample(self.negative_class,1)[0])
                # positive_text = self.get_easy_text(self.class_name)
                
            return img, positive_text, negative_text
        
        
        #! method 301 방식 
        # else:
        #     data_id = img_dir.split('/')[-1].strip('.png')
        #     positive_text = self.positive[data_id]               
        #     return img, label, positive_text 
        
        
