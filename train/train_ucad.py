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
    


def train(model, prompts, class_name, dataloader, optimizer, accelerator, log_interval: int, cfg: dict) -> dict:
    _logger.info('Train Start')
   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()    
    model.train()

    #! Train 
    for idx, (images, image_dirs) in enumerate(dataloader):

        data_time_m.update(time.time() - end)
        # predict
        output = model(images, prompts)[-1]
        loss   = model.criterion(output, image_dirs)

        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
            
        # Coreset Update for PatchCore

        losses_m.update(loss.item())            
        
        # batch time
        batch_time_m.update(time.time() - end)

        if (idx+1) % accelerator.gradient_accumulation_steps == 0:
            if ((idx+1) // accelerator.gradient_accumulation_steps) % log_interval == 0: 
                _logger.info('TRAIN [{:>4d}/{}] Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                            'LR: {lr:.3e} '
                            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            (idx+1)//accelerator.gradient_accumulation_steps, 
                            len(dataloader)//accelerator.gradient_accumulation_steps, 
                            loss       = losses_m, 
                            lr         = optimizer.param_groups[0]['lr'],
                            batch_time = batch_time_m,
                            rate       = images[0].size(0) / batch_time_m.val,
                            rate_avg   = images[0].size(0) / batch_time_m.avg,
                            data_time  = data_time_m
                            ))                
        end = time.time()    
    
    # logging metrics
    _logger.info('TRAIN: Loss: %.3f' % (losses_m.avg))
    
    train_result = {'loss' : losses_m.avg}
    return train_result 

def test(model, dataloader, class_name:str=None) -> dict:    
    from utils.metrics import MetricCalculator, loco_auroc
    
    model.eval()
    img_level = MetricCalculator(metric_list = ['auroc','average_precision'])
    pix_level = MetricCalculator(metric_list = ['auroc','average_precision'])    
    
    #! Key Identification 
    if class_name is not None:
        pass 
    else:
        test_key = [] 
        for imgs, _, labels in dataloader:
            outputs = model(imgs)[-1]
            predicted_keys = model.cpm.retrieve_key(outputs)
            test_key.extend(predicted_keys)
        key_index = pd.Series(predicted_keys).value_counts().index[0]
        class_name = model.cpm.class_names[key_index]
        
    _logger.info(f'Retrieve task result: {class_name}')
    
    #! Inference     
    for idx, (images, labels, gts) in enumerate(dataloader):

        with torch.no_grad():
            _, score, score_map = model.predict(images, class_name)
                
        # Stack Scoring for metrics 
        pix_level.update(score_map,gts.type(torch.int))
        img_level.update(score, labels.type(torch.int))
        
    # Calculate results of evaluation     
    if dataloader.dataset.name != 'CIFAR10':
        p_results = pix_level.compute()
    i_results = img_level.compute()
    
    # Calculate results of evaluation per each images        
    if dataloader.dataset.__class__.__name__ == 'MVTecLoco':
        p_results['loco_auroc'] = loco_auroc(pix_level,dataloader)
        i_results['loco_auroc'] = loco_auroc(img_level,dataloader)                
            
        
    # logging metrics
    if dataloader.dataset.name != 'CIFAR10':
        _logger.info('Image AUROC: %.3f%%| Pixel AUROC: %.3f%%' % (i_results['auroc'],p_results['auroc']))
    else:
        _logger.info('Image AUROC: %.3f%%' % (i_results['auroc']))
        
    test_result = OrderedDict(img_level = i_results)
    if dataloader.dataset.name != 'CIFAR10':
        test_result.update([('pix_level', p_results)])
    
    return test_result 


def fit(
    model, loader_dict:dict, accelerator,
    epochs: int, use_wandb: bool, log_interval: int, seed: int = None, savedir: str = None
    ,cfg=None):

    best_score = 0.0
    epoch_time_m = AverageMeter()
    end = time.time() 
    
    for class_name, class_loader_dict in loader_dict.items():
        
        #! Initialize for Training on class 
        # Dataloader
        trainloader = class_loader_dict['train']
        testloader = class_loader_dict['test']        
        # Prompts 
        prompts = model.cpm.get_prompts(class_name=class_name,device=accelerator.device) 
        # Optimizer
        optimizer = __import__('torch.optim',fromlist='optim').__dict__[cfg.OPTIMIZER.opt_name](prompts.parameters())
        # scheduler 
        if cfg.SCHEDULER.name is not None:
            scheduler = __import__('torch.optim.lr_scheduler', fromlist='lr_scheduler').__dict__[cfg.SCHEDULER.name](optimizer, **cfg.SCHEDULER.params)
        else:
            scheduler = None                
        model, optimizer, scheduler, prompts, trainloader, testloader = accelerator.prepare(model, optimizer, scheduler, prompts, trainloader,testloader)
        
        #! Key save for Task Identification
        key_pool = [] 
        with torch.no_grad():
            for imgs, img_dirs in trainloader:
                feats = model(imgs)
                key_pool.append(feats[model.cpm.num_blocks-1].detach().cpu())
                
            key_pool = torch.cat(key_pool)
        model.cpm.get_key(key_pool = key_pool, class_name = class_name)
        
        #! Training for each class 
        for step,  epoch in enumerate(range(epochs)):
            _logger.info(f'Epoch: {epoch+1}/{epochs}')
            
            train_metrics = train(
                model        = model, 
                prompts      = prompts,
                class_name   = class_name,
                dataloader   = trainloader, 
                optimizer    = optimizer, 
                accelerator  = accelerator, 
                log_interval = log_interval,
                cfg           = cfg 
            )                
            
            if scheduler is not None:
                scheduler.step()
                
            if epoch%9 == 0:
                
                #! Save Knowledge          
                
                save_knowledge(model, trainloader, prompts, class_name)       
                test_metrics = test(
                    model        = model, 
                    dataloader   = testloader,
                    class_name   = class_name
                )
                
                metric_logging(
                    savedir = savedir, use_wandb = use_wandb, epoch = epoch, step = step,
                    optimizer = optimizer, epoch_time_m = epoch_time_m,
                    train_metrics = train_metrics, test_metrics = test_metrics)        

                epoch_time_m.update(time.time() - end)
                end = time.time()
            else:
                epoch_time_m.update(time.time() - end)
                end = time.time()
                
            if best_score < test_metrics['img_level']['auroc']:
                best_score = test_metrics['img_level']['auroc']
                _logger.info(f" New best score : {best_score} | best epoch : {epoch}\n")
                # torch.save(model.state_dict(), os.path.join(savedir, f'model_best.pt')) 
        
        save_knowledge(model, trainloader, prompts, class_name)
        test_metrics = test(
                    model        = model, 
                    dataloader   = testloader,
                    class_name   = class_name
                )
                
        metric_logging(
            savedir = savedir, use_wandb = use_wandb, epoch = epoch, step = step,
            optimizer = optimizer, epoch_time_m = epoch_time_m,
            train_metrics = train_metrics, test_metrics = test_metrics,
            class_name = class_name)   
            
            
        #Todo 추후 FM, AA?와 같은 CL 관련 메트릭 추가 해야 함 
        #Todo 현재 task 가 무엇이고, test에서 검색된 task가 무엇인지 등 로깅 필요 
        #Todo Anomaly score distance 계산 하는 부분 보완 필요 
        #Todo anomaly score 리스케일링 하는 거 추가 필요 
            
            
def save_knowledge(model, trainloader, prompts, class_name):
    model.eval()
    with torch.no_grad(): 
        knowledge_pool = [] 
        for img, img_dirs in trainloader:
            # y = model(img, prompts)[-1].detach().cpu()
            y = model(img)[-1].detach().cpu()
            knowledge_pool.append(y)
        knowledge_pool = torch.cat(knowledge_pool)
        model.cpm.get_knowledge(knowledge_pool, class_name) # save sampled features by coreset method to knowledge pool 