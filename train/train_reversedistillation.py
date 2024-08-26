import logging
import time
import os 
import numpy as np
import pandas as pd

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader 
from collections import OrderedDict

from utils.metrics import MetricCalculator
from utils.log import AverageMeter,metric_logging

_logger = logging.getLogger('train')
    


def train(model, dataloader, optimizer, accelerator, log_interval: int, cfg: dict) -> dict:
    _logger.info('Train Start')
   
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    
    model.train()

    for idx, (images, _, _) in enumerate(dataloader):
        
        data_time_m.update(time.time() - end)
        # predict
        output = model(images)
        loss   = model.criterion(output)

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

def test(model, dataloader) -> dict:    
    from utils.metrics import MetricCalculator, loco_auroc
    
    model.eval()
    img_level = MetricCalculator(metric_list = ['auroc','average_precision'])
    pix_level = MetricCalculator(metric_list = ['auroc','average_precision'])
        
    for idx, (images, labels, gts) in enumerate(dataloader):

        with torch.no_grad():
            outputs = model(images)   
            score_map = model.get_score_map(outputs).detach().cpu()
            score = score_map.reshape(score_map.shape[0],-1).max(-1)[0]
                
        # Stack Scoring for metrics 
        pix_level.update(score_map,gts.type(torch.int))
        img_level.update(score, labels.type(torch.int))
        
    # Calculate results of evaluation     
    if dataloader.dataset.name != 'CIFAR10':
        p_results = pix_level.compute()
    i_results = img_level.compute()    
                    
    # logging metrics
    if dataloader.dataset.name != 'CIFAR10':
        _logger.info('Image AUROC: %.3f%%| Pixel AUROC: %.3f%%' % (i_results['auroc'],p_results['auroc']))
    else:
        _logger.info('Image AUROC: %.3f%%' % (i_results['auroc']))
        
    test_result = OrderedDict(img_level = i_results)
    if dataloader.dataset.name != 'CIFAR10':
        test_result.update([('pix_level', p_results)])
    
    return test_result 

def task_inference(model, scenario, accelerator, cfg):
    
    
    if scenario.nb_tasks == 1:
        range_ = range(0,1)
    else:
        range_ = range(0,scenario.nb_tasks-1)
        
    j_task_result_dict = {} 
    for j in range_:
        trainset, _,_,_ = scenario(j,model)
        trainloader = accelerator.prepare(DataLoader(dataset = trainset, batch_size  = cfg.DATASET.batch_size, num_workers = cfg.DATASET.num_workers, shuffle = True))
        
        # j task inference on t model 
        j_task_result = [] 
        with torch.no_grad():
            for idx, (images, _, _) in enumerate(trainloader):
                outputs = model(images)   
                score_map = model.get_score_map(outputs).detach().cpu().numpy()
                j_task_result.append(score_map)                
        j_task_result = np.mean(np.concatenate(j_task_result))
        
        j_task_result_dict[j] = j_task_result 
    return j_task_result_dict 

def fit(
    model, scenario, testloader, accelerator,
    epochs: int, use_wandb: bool, log_interval: int, seed: int = None, savedir: str = None
    ,cfg=None):

    best_score = 0.0
    epoch_time_m = AverageMeter()
    end = time.time() 
    
    for t in range(scenario.nb_tasks):
        
        trainset, optimizer, scheduler, epochs = scenario(t,model)
        trainloader = DataLoader(
                dataset     = trainset,
                batch_size  = cfg.DATASET.batch_size,
                num_workers = cfg.DATASET.num_workers,
                shuffle     = True 
            )   
        
        model, trainloader, testloader, optimizer, scheduler = accelerator.prepare(model, trainloader,testloader, optimizer, scheduler)
    
        for  epoch in range(epochs):
            _logger.info(f'Epoch: {epoch+1}/{epochs}')
            
            train_metrics = train(
                model        = model, 
                dataloader   = trainloader, 
                optimizer    = optimizer, 
                accelerator  = accelerator, 
                log_interval = log_interval,
                cfg           = cfg 
            )
            
            if scheduler is not None:
                scheduler.step()
                
            if epoch%49 == 0:
                test_metrics = test(
                        model        = model, 
                        dataloader   = testloader
                    )
                
                epoch_time_m.update(time.time() - end)
                end = time.time()
                metric_logging(
                    savedir = savedir, use_wandb = use_wandb, epoch = epoch, step = epoch,
                    optimizer = optimizer, epoch_time_m = epoch_time_m, task=t,
                    train_metrics = train_metrics, test_metrics = test_metrics
                    )  
            
        # Evaluation t model
        test_metrics = test(
            model        = model, 
            dataloader   = testloader
        )
        scenario.auc_result[t] = test_metrics
        
        # Evaluation t model on j task                 
        j_task_result_dict = task_inference(model, scenario, accelerator, cfg)
        scenario.fm_result[t] = j_task_result_dict
        
        # save model parameter at t task 
        scenario.task_model_list.append(model.parameters()) 
            
        epoch_time_m.update(time.time() - end)
        end = time.time()
        
        metric_logging(
            savedir = savedir, use_wandb = use_wandb, epoch = epoch, step = epoch,
            optimizer = optimizer, epoch_time_m = epoch_time_m, task=t,
            train_metrics = train_metrics, test_metrics = test_metrics)        

    
    # Evaluation for Continual learning 
    scenario_result = scenario.eval()
    metric_logging(
        savedir = savedir, 
        task = -1,
        test_metrics = scenario_result
    )