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
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    con_losses_m = AverageMeter()
    neg_losses_m = AverageMeter()
    
    end = time.time()
    
    model.train()
    for idx, (images, positive, negative) in enumerate(dataloader):
        data_time_m.update(time.time() - end)
        
        # predict        
        loss_dict = model(images, positive, negative, prompts)
        loss = loss_dict['total_loss']
        con_loss = loss_dict['contrastive_loss']
        neg_loss = loss_dict['negative_loss']
        
        # loss backward
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        
        # loss logging 
        losses_m.update(loss.item())    
        con_losses_m.update(con_loss.item())    
        neg_losses_m.update(neg_loss.item())    
        
        # batch time
        batch_time_m.update(time.time() - end)

        
    log = ('[{:d}/{}] Total Loss: {loss.avg:>6.4f} Cont Loss: {con_loss.avg:>6.4f} Neg Loss: {neg_loss.avg:>6.4f} '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.avg:.3f}s, {rate_avg:>3.2f}/s '
                        'Data: {data_time.avg:.3f}s'.format(
                        (idx+1)//accelerator.gradient_accumulation_steps, 
                        len(dataloader)//accelerator.gradient_accumulation_steps, 
                        loss       = losses_m, 
                        con_loss   = con_losses_m, 
                        neg_loss   = neg_losses_m, 
                        lr         = optimizer.param_groups[0]['lr'],
                        batch_time = batch_time_m,
                        rate_avg   = images[0].size(0) / batch_time_m.avg,
                        data_time  = data_time_m
                        ))     
    end = time.time()    
    return log 

def test(model, prompts, dataloader, device, 
         savedir, use_wandb, epoch, optimizer, epoch_time_m, class_name, current_class_name,
          knowledge=None, agnostic=False, last=False) -> dict:
    from utils.metrics import MetricCalculator, loco_auroc
    
    model.eval()
    img_level = MetricCalculator(metric_list = ['auroc','average_precision'])
    pix_level = MetricCalculator(metric_list = ['auroc','average_precision'])     

    #! Knowledge 
    model.anomaly_scorer.fit(knowledge)
    #! Inference     
    for idx, (images, labels, gts) in enumerate(dataloader):

        with torch.no_grad():
            if agnostic:
                features = model.embed_img(images).detach().cpu().numpy()
                query_features = np.mean(features,axis=(0,1))
                prompts = model.pool.retrieve_prompts(prompts, query_features).to(device)
            
            _, score, score_map = model.predict(images, prompts)
                
        # Stack Scoring for metrics 
        pix_level.update(score_map,gts.type(torch.int))
        img_level.update(score, labels.type(torch.int))
            
    i_results, p_results = img_level.compute(), pix_level.compute()
    _logger.info(f"Class name : {current_class_name} Image AUROC: {i_results['auroc']:.3f}| Pixel AUROC: {p_results['auroc']:.3f}")
        
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

def create_knowledge(model, prompts, trainloader,prompts_save:bool=False):
    '''
            Description 
                - 학습 완료 후 현재 Task의 feature를 추출한 다음 Knowledge Pool에 저장 
                - feature 추출 -> coreset sampling -> knowledge pool에 extend 
            Return 
                - 모든 feature 저장되어 있는 knowledge pool
                - 현재 task의 feature 
        ''' 
    features_bank = [] 
    model.eval()
    for idx, (images, _, _) in enumerate(trainloader):
        features = model.embed_img(images, prompts)
        features_bank.append(features.detach().cpu().numpy())
        
    features_bank = np.concatenate(features_bank)
    sampled_features = model.fit(features_bank) # sampling feature to save in memory bank 
    
    # knowledge & key save 
    class_name = trainloader.dataset.class_name
    knowledge_pool = model.pool.get_knowledge(class_name = class_name, 
                                                knowledge  = sampled_features)
    _logger.info(f"knowledge 크기 : {len(knowledge_pool)}")
    
    # prompts save    
    if prompts_save: 
        num_layer = prompts.num_layers[0]
        model.pool.prompts.extend(
            prompts[str(num_layer)].detach().cpu().numpy()
        )
    return knowledge_pool, sampled_features


def fit(
    model, loader_dict:dict, accelerator,
    epochs: int, use_wandb: bool, log_interval: int, eval_interval: int, seed: int = None, savedir: str = None
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
            # train one epoch 
            if n_task !=0:
                train_log = train(
                    model        = model, 
                    prompts      = prompts, 
                    dataloader   = trainloader, 
                    optimizer    = optimizer, 
                    scheduler    = scheduler,
                    accelerator  = accelerator, 
                    log_interval = log_interval,
                    cfg           = cfg 
                )                
                if scheduler:
                    scheduler.step()
                
                epoch_time_m.update(time.time() - end)
                end = time.time()
                
                # logging 
                if ((epoch+1) % log_interval)== 0: 
                    _logger.info(f"Train Epoch [{epoch}/{epochs}] " + train_log)
                    
            # Evaluation 
            if ((epoch+1) % eval_interval)== 0: 
                
                # Create knowledge and save         
                knowledge_pool, class_features = create_knowledge(model, prompts, trainloader, prompts_save=False)
                
                # Task-specific / agnostic inference 
                for agnostic in [False]:
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
                                knowledge          = [class_features],
                                agnostic           = agnostic,
                                last               = False
                            )
                
        # Create knowledge and save         
        knowledge_pool, class_features = create_knowledge(model, prompts, trainloader, prompts_save=True)
        
        # Task-agnostic inference 
        for agnostic in [True]:
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
                        knowledge          = [class_features] if agnostic else np.expand_dims(np.array(model.pool.knowledge),0),
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
            epoch              = 0 if epochs == 0 else epoch,
            optimizer          = optimizer, 
            epoch_time_m       = epoch_time_m,
            class_name         = class_name,
            current_class_name = class_name,
            dataloader         = testloader,
            knowledge          = np.expand_dims(np.array(model.pool.knowledge),0),
            agnostic           = True,
            last               = True
        )