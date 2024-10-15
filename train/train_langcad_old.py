import logging
import time
import os 
import numpy as np
import pandas as pd

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from collections import OrderedDict

from utils.metrics import MetricCalculator
from utils.log import AverageMeter,metric_logging

_logger = logging.getLogger('train')


def train(model, dataloader, optimizer, accelerator, log_interval: int, cfg) -> dict:
    print('Train Start')
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    
    model.train()
    train_bank = [] 
    for idx, (images, _, _) in enumerate(dataloader):
        data_time_m.update(time.time() - end)
        # predict
        
        img_features = model.embed(images, cls_name=cfg.DATASET.class_name) 
        train_bank.append(img_features.detach().cpu().numpy())    
        
        # batch time
        batch_time_m.update(time.time() - end)

        if ((idx+1) // accelerator.gradient_accumulation_steps) % log_interval == 0: 
            _logger.info('TRAIN [{:>4d}/{}]'
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        (idx+1)//accelerator.gradient_accumulation_steps, 
                        len(dataloader)//accelerator.gradient_accumulation_steps, 
                        batch_time = batch_time_m,
                        rate       = images[0].size(0) / batch_time_m.val,
                        rate_avg   = images[0].size(0) / batch_time_m.avg,
                        data_time  = data_time_m
                        ))                
        end = time.time()
        
    train_bank = np.concatenate(train_bank)
    model.fit(train_bank)

def test(model, dataloader,cfg) -> dict:    
    from utils.metrics import MetricCalculator, loco_auroc        
    model.eval()
    img_level = MetricCalculator(metric_list = ['auroc','average_precision'])
    pix_level = MetricCalculator(metric_list = ['auroc','average_precision'])
        
    for idx, (images, labels, gts) in enumerate(dataloader):
        
        # predict
        score, img_score, pixel_score = model.predict(images,cls_name=cfg.DATASET.class_name)
                
        # Stack Scoring for metrics 
        pix_level.update(pixel_score,gts.type(torch.int))
        img_level.update(img_score, labels.type(torch.int))                  
            
    # Compute evaluation result 
    p_results = pix_level.compute()
    i_results = img_level.compute()            
    
    # logging metrics
    _logger.info('Image AUROC: %.3f%%| Pixel AUROC: %.3f%%' % (i_results['auroc'],p_results['auroc']))

        
    test_result = OrderedDict(img_level = i_results)
    test_result.update([('pix_level', p_results)])
    
    return test_result 


def fit(
    model, trainloader, testloader, optimizer, scheduler, accelerator,
    epochs: int, use_wandb: bool, log_interval: int, seed: int = None, savedir: str = None
    ,cfg=None):

    best_score = 0.0
    epoch_time_m = AverageMeter()
    end = time.time() 
    
    #! 임시 보조 장치 
    train(
        model        = model, 
        dataloader   = trainloader, 
        optimizer    = optimizer, 
        accelerator  = accelerator, 
        log_interval = log_interval,
        cfg          = cfg 
    )        
    
    test_metrics = test(
        model        = model, 
        dataloader   = testloader,
        cfg          = cfg 
    )
    
    epoch_time_m.update(time.time() - end)
    end = time.time()
    
    
    # logging 
    metric_logging(
        savedir = savedir, use_wandb = use_wandb, epoch = 1, step = 1,
        optimizer = optimizer, epoch_time_m = epoch_time_m,
        train_metrics = None, test_metrics = test_metrics)
        
                
    # checkpoint - save best results and model weights        
    if best_score < test_metrics['img_level']['auroc']:
        best_score = test_metrics['img_level']['auroc']
        print(f" New best score : {best_score} | best epoch : {1}\n")
        

