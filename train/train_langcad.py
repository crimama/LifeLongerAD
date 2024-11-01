import logging
import time
import os 
import numpy as np
import pandas as pd

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from collections import OrderedDict
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from utils.metrics import MetricCalculator
from utils.log import AverageMeter,metric_logging
import warnings
warnings.filterwarnings('ignore')

_logger = logging.getLogger('train')
    


def train(model, prompts, dataloader, optimizer, scheduler, accelerator, log_interval: int, cfg) -> dict:
    print('Train Start')
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    
    end = time.time()
    
    model.train()
    used_memory = log_vram()
    _logger.info(f'current memory : {used_memory}')
    for idx, (images, positive, negative) in enumerate(dataloader):
        data_time_m.update(time.time() - end)
        
        # predict        
        loss = model(images, positive, negative, prompts)
        
        # loss backward
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        losses_m.update(loss.item())    
        
        # batch time
        batch_time_m.update(time.time() - end)

        
        if ((idx+1) % log_interval )== 0: 
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
    
    used_memory = log_vram()
    _logger.info(f'train one epoch done current memory : {used_memory}')    
    
def test(model, prompts, dataloader, class_name:str=None, knowledge=None) -> dict:
    from utils.metrics import MetricCalculator, loco_auroc
    
    model.eval()
    img_level = MetricCalculator(metric_list = ['auroc','average_precision'])
    pix_level = MetricCalculator(metric_list = ['auroc','average_precision'])     

    #! Knowledge 
    model.anomaly_scorer.fit([knowledge])    
    #! Inference     
    for idx, (images, labels, gts) in enumerate(dataloader):

        with torch.no_grad():
            _, score, score_map = model.predict(images, prompts)
                
        # Stack Scoring for metrics 
        pix_level.update(score_map,gts.type(torch.int))
        img_level.update(score, labels.type(torch.int))
            
    i_results, p_results = img_level.compute(), pix_level.compute()
    _logger.info('Image AUROC: %.3f%%| Pixel AUROC: %.3f%%' % (i_results['auroc'],p_results['auroc']))
        
    test_result = OrderedDict(img_level = i_results)
    test_result.update([('pix_level', p_results)])    
    return test_result 

    
def test_agnostic(model, prompts, dataloader, device, class_name:str=None, knowledge=None) -> dict:
    from utils.metrics import MetricCalculator, loco_auroc
    
    model.eval()
    img_level = MetricCalculator(metric_list = ['auroc','average_precision'])
    pix_level = MetricCalculator(metric_list = ['auroc','average_precision'])     

    #! Knowledge 
    model.anomaly_scorer.fit([knowledge])    
    #! Inference     
    for idx, (images, labels, gts) in enumerate(dataloader):

        with torch.no_grad():
            features = model.embed(images).detach().cpu().numpy()
            query_features = np.mean(features,axis=(0,1))

            prompts = model.pool.retrieve_prompts(prompts, query_features).to(device)
            
            _, score, score_map = model.predict(images, prompts)
                
        # Stack Scoring for metrics 
        pix_level.update(score_map,gts.type(torch.int))
        img_level.update(score, labels.type(torch.int))
            
    i_results, p_results = img_level.compute(), pix_level.compute()
    _logger.info('Image AUROC: %.3f%%| Pixel AUROC: %.3f%%' % (i_results['auroc'],p_results['auroc']))
        
    test_result = OrderedDict(img_level = i_results)
    test_result.update([('pix_level', p_results)])    
    return test_result 

def create_knowledge(model, prompts, trainloader):
    features_bank = [] 
    model.eval()
    for idx, (images, _, _) in enumerate(trainloader):
        features = model.embed(images, prompts, out_with_prompts=False)
        features_bank.append(features.detach().cpu().numpy())
        
    features_bank = np.concatenate(features_bank)
    sampled_features = model.fit(features_bank) # sampling feature to save in memory bank 
    
    # knowledge & key save 
    class_name = trainloader.dataset.class_name
    knowledge_pool = model.pool.get_knowledge(class_name = class_name , 
                                            knowledge  = sampled_features)
    _logger.info(f"knowledge 크기 : {len(knowledge_pool)}")
    # prompts save 
    # model.pool.prompts[class_name] = prompts.to('cpu')
    model.pool.prompts.extend(
        prompts['3'].detach().cpu().numpy()
    )
    return knowledge_pool, sampled_features


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
        
        # Init prompt 
        prompts = model.create_prompts()
        
        # Init optimzier & SCheduler 
        optimizer = __import__('torch.optim',fromlist='optim').__dict__[cfg.OPTIMIZER.opt_name](prompts.parameters(), lr=cfg.OPTIMIZER.lr, **cfg.OPTIMIZER.params)        
        if cfg.SCHEDULER.name is not None:
            scheduler = __import__('torch.optim.lr_scheduler', fromlist='lr_scheduler').__dict__[cfg.SCHEDULER.name](optimizer, **cfg.SCHEDULER.params)
        else:
            scheduler = None
        
        # scheduler = CosineAnnealingWarmupRestarts(optimizer,
        #                                     first_cycle_steps=10,
        #                                     cycle_mult=1.0,
        #                                     max_lr=0.1,
        #                                     min_lr=0.001,
        #                                     warmup_steps=5,
        #                                     gamma=1.0)
            
        # Init Dataloader 
        trainloader, testloader = loader_dict[class_name]['train'],loader_dict[class_name]['test']
        
        model, prompts, trainloader, testloader, optimizer, scheduler = accelerator.prepare(model, prompts, trainloader, testloader, optimizer, scheduler)
        
        # Train 
        if i == 0:
            epoch = 0
            step = 0 
            epoch_time_m.update(time.time() - end)
            end = time.time()
        else:
            for step,  epoch in enumerate(range(epochs)):
                _logger.info(f'Epoch: {epoch+1}/{epochs}')
                # train one epoch 
                train(
                    model        = model, 
                    prompts      = prompts, 
                    dataloader   = trainloader, 
                    optimizer    = optimizer, 
                    scheduler    = scheduler,
                    accelerator  = accelerator, 
                    log_interval = log_interval,
                    cfg           = cfg 
                )                

                epoch_time_m.update(time.time() - end)
                end = time.time()
            
            # scheduler.step()
        # Create knowledge and save 
        knowledge_pool, class_features = create_knowledge(model, prompts, trainloader)
                
        # Task-specific inference 
        test_metrics = test(
                    model        = model, 
                    prompts      = prompts.to(accelerator.device), 
                    dataloader   = testloader,
                    knowledge    = class_features
                )
        metric_logging(
            savedir = savedir, use_wandb = use_wandb, epoch = epoch, step = step,
            optimizer = optimizer, epoch_time_m = epoch_time_m,
            test_metrics = test_metrics,
            class_name = class_name, current_class_name = class_name,
            **{'task_agnostic' : 'specific'}
            ) 
        
        # Task-agnostic inference 
        test_metrics = test_agnostic(
                    model        = model, 
                    prompts      = prompts, 
                    device       = accelerator.device,
                    dataloader   = testloader,
                    knowledge    = np.array(knowledge_pool) 
                )
        
        metric_logging(
            savedir = savedir, use_wandb = use_wandb, epoch = epoch, step = step,
            optimizer = optimizer, epoch_time_m = epoch_time_m,
            test_metrics = test_metrics,
            class_name = 'None', current_class_name = class_name,
            **{'task_agnostic' : 'agnostic'}
            )  

    # Pool save 
    model.pool.save_pool(
        save_path = os.path.join(savedir,'last_pool.pth')
    )
    
    # agnostic last 
    for i, (class_name, class_loader_dict) in enumerate(loader_dict.items()):
        testloader = loader_dict[class_name]['test']
        testloader = accelerator.prepare(testloader)
        
        test_metrics = test_agnostic(
            model        = model, 
            prompts      = prompts, 
            device       = accelerator.device,
            dataloader   = testloader,
            knowledge    = np.array(knowledge_pool)  
        )
        
        metric_logging(
            savedir = savedir, use_wandb = use_wandb, epoch = epoch, step = step,
            optimizer = optimizer, epoch_time_m = epoch_time_m,
            test_metrics = test_metrics,
            class_name = 'None', current_class_name = class_name,
            **{'task_agnostic' : 'agnostic-last'}
            )  
        

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