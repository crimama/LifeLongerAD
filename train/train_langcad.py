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

def test(model, prompts, dataloader, device, 
         savedir, use_wandb, epoch, optimizer, epoch_time_m, class_name, current_class_name,
          knowledge=None, agnostic=False, last=False) -> dict:
    from utils.metrics import MetricCalculator, loco_auroc
    
    model.eval()
    img_level = MetricCalculator(metric_list = ['auroc','average_precision'])
    pix_level = MetricCalculator(metric_list = ['auroc','average_precision'])     

    #! Knowledge 
    model.anomaly_scorer.fit([knowledge])    
    #! Inference     
    for idx, (images, labels, gts) in enumerate(dataloader):

        with torch.no_grad():
            if agnostic:
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
    
    #! Metric Logging 
    if agnostic:
        if last:
            agnostic = 'agnostic-last'
        else:
            agnostic = 'agnostic'
    else:
        agnostic = 'specific'
    
    metric_logging(
            savedir = savedir, use_wandb = use_wandb, epoch = epoch,
            optimizer = optimizer, epoch_time_m = epoch_time_m,
            test_metrics = test_result,
            class_name = 'None', current_class_name = class_name,
            **{'task_agnostic' : agnostic}
            )
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
    knowledge_pool = model.pool.get_knowledge(class_name = class_name, 
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
    
    for n_task, (class_name, class_loader_dict) in enumerate(loader_dict.items()):
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
            
        # Init Dataloader 
        trainloader, testloader = loader_dict[class_name]['train'],loader_dict[class_name]['test']
        
        model, prompts, trainloader, testloader, optimizer, scheduler = accelerator.prepare(model, prompts, trainloader, testloader, optimizer, scheduler)
        
        # Train 
        for epoch in range(epochs):
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
            
            if scheduler:
                scheduler.step()
                
        # Create knowledge and save 
        '''
            Description 
                - 학습 완료 후 현재 Task의 feature를 추출한 다음 Knowledge Pool에 저장 
                - feature 추출 -> coreset sampling -> knowledge pool에 extend 
            Return 
                - 모든 feature 저장되어 있는 knowledge pool
                - 현재 task의 feature 
        ''' 
        knowledge_pool, class_features = create_knowledge(model, prompts, trainloader)
                
        # Task-specific / agnostic inference 
        for agnostic in [False, True]:
            test_metrics = test(
                        model              = model, 
                        prompts            = prompts.to(accelerator.device), 
                        device             = accelerator.device,
                        savedir            = savedir, 
                        use_wandb          = use_wandb,
                        epoch              = 0 if epochs == 0 else epoch,
                        optimizer          = optimizer, 
                        epoch_time_m       = epoch_time_m,
                        class_name         = class_name,
                        current_class_name = class_name,
                        dataloader         = testloader,
                        knowledge          = class_features,
                        agnostic           = agnostic,
                        last               = False
                    )

    # Pool save 
    model.pool.save_pool(
        save_path = os.path.join(savedir,'last_pool.pth')
    )
    
    # agnostic last 
    for i, (class_name, class_loader_dict) in enumerate(loader_dict.items()):
        testloader = loader_dict[class_name]['test']
        testloader = accelerator.prepare(testloader)
        
        test_metrics = test(
            model              = model, 
            prompts            = prompts.to(accelerator.device), 
            device             = accelerator.device,
            savedir            = savedir, 
            use_wandb          = use_wandb,
            epoch              = epoch,
            optimizer          = optimizer, 
            epoch_time_m       = epoch_time_m,
            class_name         = class_name,
            current_class_name = class_name,
            dataloader         = testloader,
            knowledge          = class_features,
            agnostic           = True,
            last               = True
        )