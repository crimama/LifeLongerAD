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
    used_memory = log_vram()
    _logger.info(f'current memory : {used_memory}')
    for idx, (images, _, _) in enumerate(dataloader):
        data_time_m.update(time.time() - end)
        # predict
        
        img_features = model.embed(images) 
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
    
    used_memory = log_vram()
    _logger.info(f'embed done current memory : {used_memory}')
    
    train_bank = np.concatenate(train_bank)
    sampled_features = model.fit(train_bank)
    
    used_memory = log_vram()
    _logger.info(f'fit done current memory : {used_memory}')
    
    model.pool.get_knowledge(class_name = dataloader.dataset.class_name, 
                             knowledge  = sampled_features)# knowledge save 
    
    used_memory = log_vram()
    _logger.info(f'get knowledge done current memory : {used_memory}')
    return sampled_features
    
def test(model, dataloader, class_name:str=None, knowledge=None) -> dict:
    from utils.metrics import MetricCalculator, loco_auroc
    
    model.eval()
    img_level = MetricCalculator(metric_list = ['auroc','average_precision'])
    pix_level = MetricCalculator(metric_list = ['auroc','average_precision'])     

    #! Knowledge 
    model.anomaly_scorer.fit([knowledge])
    #! Inference     
    for idx, (images, labels, gts) in enumerate(dataloader):

        with torch.no_grad():
            _, score, score_map = model.predict(images)
                
        # Stack Scoring for metrics 
        pix_level.update(score_map,gts.type(torch.int))
        img_level.update(score, labels.type(torch.int))
            
    i_results, p_results = img_level.compute(), pix_level.compute()
    _logger.info('Image AUROC: %.3f%%| Pixel AUROC: %.3f%%' % (i_results['auroc'],p_results['auroc']))
        
    test_result = OrderedDict(img_level = i_results)
    test_result.update([('pix_level', p_results)])    
    return test_result 


def fit(
    model, loader_dict:dict, accelerator,
    epochs: int, use_wandb: bool, log_interval: int, seed: int = None, savedir: str = None
    ,cfg=None):

    best_score = 0.0
    epoch_time_m = AverageMeter()
    end = time.time() 
    
    for i, (class_name, class_loader_dict) in enumerate(loader_dict.items()):
        torch.cuda.empty_cache()
        _logger.info(f"Current Class Name : {class_name}")
        optimizer = None 
        scheduler = None 
        trainloader,testloader = loader_dict[class_name]['train'],loader_dict[class_name]['test']
        
        model, trainloader, testloader = accelerator.prepare(model, trainloader, testloader)
        for step,  epoch in enumerate(range(epochs)):
            _logger.info(f'Epoch: {epoch+1}/{epochs}')
            knowledge = train(
                model        = model, 
                dataloader   = trainloader, 
                optimizer    = None, 
                accelerator  = accelerator, 
                log_interval = log_interval,
                cfg           = cfg 
            )                

            epoch_time_m.update(time.time() - end)
            end = time.time()
            
        #current inference 
        test_metrics = test(
                    model        = model, 
                    dataloader   = testloader,
                    knowledge    = knowledge
                )
        used_memory = log_vram()
        metric_logging(
            savedir = savedir, use_wandb = use_wandb, epoch = epoch, step = step,
            optimizer = optimizer, epoch_time_m = epoch_time_m,
            test_metrics = test_metrics,
            class_name = class_name, current_class_name = class_name,
            **{'task_agnostic' : 'specific'}
            )    
        
        # Task-agnostic inference 
        ta_class_name = model.pool.retrieve_key(model, testloader)
        knowledge = model.pool.knowledge[ta_class_name]
        
        test_metrics = test(
                    model        = model, 
                    dataloader   = testloader,
                    knowledge    = knowledge 
                )
        
        metric_logging(
            savedir = savedir, use_wandb = use_wandb, epoch = epoch, step = step,
            optimizer = optimizer, epoch_time_m = epoch_time_m,
            test_metrics = test_metrics,
            class_name = ta_class_name, current_class_name = class_name,
            **{'task_agnostic' : 'agnostic'}
            )   
        
    for i, (class_name, class_loader_dict) in enumerate(loader_dict.items()):
        testloader = loader_dict[class_name]['test']
        testloader = accelerator.prepare(testloader)
        
        ta_class_name = model.pool.retrieve_key(model, testloader)
        knowledge = model.pool.knowledge[ta_class_name]
        
        test_metrics = test(
            model        = model, 
            dataloader   = testloader,
            knowledge    = knowledge 
        )
        
        metric_logging(
            savedir = savedir, use_wandb = use_wandb, epoch = epoch, step = step,
            optimizer = optimizer, epoch_time_m = epoch_time_m,
            test_metrics = test_metrics,
            class_name = ta_class_name, current_class_name = class_name,
            **{'task_agnostic' : 'agnostic-last'}
            )

#Todo 1. Key 인식 - np.mean(query features) vs pool.key -> 해결 
#Todo 2. current inference -> 해결 
#Todo 3. task-agnostic inference 

def log_vram():
    import pynvml

    # NVML 초기화
    pynvml.nvmlInit()

    # GPU 개수 확인
    device_count = pynvml.nvmlDeviceGetCount()

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
    pynvml.nvmlShutdown()
    
    return (info.used / 1024 ** 2)
        
import sys
import pdb

def excepthook(type, value, traceback):
    # 에러가 발생하면 PDB 디버거를 호출합니다.
    pdb.post_mortem(traceback)