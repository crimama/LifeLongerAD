import wandb 
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
from utils.log import AverageMeter,metric_logging,DriftMonitor
import warnings
warnings.filterwarnings('ignore')

_logger = logging.getLogger('train')
    

def train(model, dataloader, testloader, optimizer, scheduler, accelerator, log_interval: int, epoch, epochs, savedir, cfg, drift_monitor) -> dict:
    
    def collect_gradients(cfg, model, all_gradients, epoch, step):
        if ((cfg.CONTINUAL.online and (step % 10 == 0)) or (not cfg.CONTINUAL.online and ((epoch) % 2 == 0) and (step % 4 == 0))):
            step_grad_dict = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    step_grad_dict[name] = param.grad.clone().detach().cpu().numpy()
            all_gradients.append(step_grad_dict)
    
    def log_training_info(step, accelerator, dataloader, epoch, epochs,
                      losses_m, feature_losses_m, svd_losses_m,
                      optimizer, batch_time_m, data_time_m, images, wandb_use:bool = False):
        current_step = (step + 1) // accelerator.gradient_accumulation_steps
        total_steps = len(dataloader) // accelerator.gradient_accumulation_steps
        _logger.info(
            'Train Epoch [{epoch}/{epochs}] [{current_step:d}/{total_steps:d}] '
            'Total Loss: {loss_val:>6.4f} | '
            'Feature Loss: {feature_loss_val:>6.4f} | '            
            'SVD Loss: {svd_loss_val:>6.4f} | '
            'LR: {lr:.3e} | '
            'Time: {batch_time_avg:.3f}s, {rate_avg:>3.2f}/s | '
            'Data: {data_time_avg:.3f}s'.format(
                current_step=current_step,
                total_steps=total_steps,
                epoch=epoch,
                epochs=epochs,
                loss_val=losses_m.val,
                feature_loss_val=feature_losses_m.val,
                svd_loss_val=svd_losses_m.val,
                lr=optimizer.param_groups[0]['lr'],
                batch_time_avg=batch_time_m.avg,
                rate_avg=images[0].size(0) / batch_time_m.avg,
                data_time_avg=data_time_m.avg
            )
        )
        
        if wandb_use:
            wandb.log(
                {
                'Train/Epoch': epoch,  # 현재 에포크 로깅 (선택 사항)
                'Train/Total Loss': losses_m.avg, # 평균 손실 로깅
                'Train/Feature Loss': feature_losses_m.avg,
                'Train/SVD Loss': svd_losses_m.avg,
                'Train/Learning Rate': optimizer.param_groups[0]['lr'],
                'Time/Train Batch Average (s)': batch_time_m.avg,
                'Time/Processing Rate (img/s)': images[0].size(0) / batch_time_m.avg,
                'Time/Data Loading Average (s)': data_time_m.avg,
                'Train/Total Loss (val)': losses_m.val,
                'Train/Feature Loss (val)': feature_losses_m.val,
                'Train/SVD Loss (val)': svd_losses_m.val,
            }
        )

    
    def do_online_inference(cfg, step, dataloader, model, accelerator, savedir, epoch, testloader, current_class_name, batch_time_m, optimizer):
        if ((cfg.CONTINUAL.online and (step % 10 == 0)) or (step == len(dataloader) - 1)):
            test_metrics = test(
                model=model, device=accelerator.device, savedir=savedir, use_wandb=False,
                epoch=step if cfg.CONTINUAL.online else step*epoch, optimizer=optimizer,
                epoch_time_m=batch_time_m, class_name=current_class_name,
                current_class_name=current_class_name, dataloader=testloader
            )
    
    def save_gradients_if_needed(cfg, dataloader, epoch, savedir, all_gradients):
        if (cfg.CONTINUAL.online or (not cfg.CONTINUAL.online and ((epoch) % 2 == 0))):
            current_class_name_ = dataloader.dataset.class_name
            np.save(f"{savedir}/gradients/{current_class_name_}_gradient_log_epoch_{epoch}.npy", all_gradients)
    
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    
    losses_m = AverageMeter()
    feature_losses_m = AverageMeter()
    svd_losses_m = AverageMeter()
    
    current_class_name = dataloader.dataset.class_name
    model.train()
    all_gradients = []
    
    end = time.time()
    for step, (images, labels, class_labels) in enumerate(dataloader):
        Input = {'image':images,'clslabel':class_labels}
        data_time_m.update(time.time() - end)
        
        # Training 
        outputs = model(Input) 
        loss = model.criterion(outputs, Input)
        optimizer.zero_grad()
        accelerator.backward(loss['loss'])         
        
        # Loss record  
        losses_m.update(loss['loss'].item())
        feature_losses_m.update(loss['feature_loss'])
        svd_losses_m.update(loss['svd_loss'])
        
        #* collect_gradients(cfg, model, all_gradients, epoch, step)
        drift_monitor.update(outputs['feature_rec'][0])
        optimizer.step()
        
        batch_time_m.update(time.time() - end)        
        # Logging 
        adjusted_log_interval = log_interval if cfg.CONTINUAL.online else 1
        if (step + 1) % adjusted_log_interval == 0:
            log_training_info(step, accelerator, dataloader, epoch, epochs, 
                            losses_m ,feature_losses_m ,svd_losses_m,
                            optimizer, batch_time_m, data_time_m, images, wandb_use=cfg.TRAIN.wandb.use)
            
        # do_online_inference(cfg, step, dataloader, model, accelerator, savedir, epoch, testloader, current_class_name, batch_time_m, optimizer)
        end = time.time()
    
    return {"loss": losses_m.avg, "gradients": all_gradients, "class_name": current_class_name}

def test(model, dataloader, device, 
         savedir, use_wandb, epoch, optimizer, epoch_time_m, class_name, current_class_name,
         last : bool = False) -> dict:
    from utils.metrics import MetricCalculator, loco_auroc    
    model.eval()
    img_level = MetricCalculator(metric_list = ['auroc','average_precision'])
    pix_level = MetricCalculator(metric_list = ['auroc','average_precision'])     

    test_time_m = AverageMeter()    
    
    #! Inference     
    end = time.time()
    for idx, (images, labels, class_labels, gts) in enumerate(dataloader):

        with torch.no_grad():
            Input = {'image':images,'clsname':class_labels}
            outputs = model(Input)   
            score_map = outputs['pred'].detach().cpu()            
            score = score_map.reshape(score_map.shape[0],-1).max(-1)[0]
                
        # Stack Scoring for metrics 
        pix_level.update(score_map,gts.type(torch.int))
        img_level.update(score, labels.type(torch.int))
        
        test_time_m.update(time.time() - end)
        end = time.time()
            
    i_results, p_results = img_level.compute(), pix_level.compute()
    _logger.info(f"Current Class name : {current_class_name} Class name : {class_name} Image AUROC: {i_results['auroc']:.3f}| Pixel AUROC: {p_results['auroc']:.3f}")
        
    test_result = OrderedDict(img_level = i_results)
    test_result.update([('pix_level', p_results)])
    
    metric_logging(
            savedir = savedir, use_wandb = use_wandb, epoch = epoch,
            optimizer = optimizer, epoch_time_m = test_time_m,
            test_metrics = test_result,
            class_name = class_name, current_class_name = current_class_name,
            **{'last' : last}
            )
    return test_result 


def fit(
    model, loader_dict:dict, accelerator,
    epochs: int, use_wandb: bool, log_interval: int, eval_interval: int, seed: int = None, savedir: str = None
    ,cfg=None):
    print(savedir)
    best_score = 0.0
    epoch_time_m = AverageMeter()
    end = time.time() 
    
    
    for n_task, (current_class_name, class_loader_dict) in enumerate(loader_dict.items()):
        if (n_task == 0) or (cfg.CONTINUAL.continual==False):
            drift_monitor = DriftMonitor(log_dir=os.path.join(savedir,'DriftMonitor.log'))
        
        torch.cuda.empty_cache()
        _logger.info(f"Current Class Name : {current_class_name}")        
            
        # Init optimzier & SCheduler 
        optimizer = __import__('torch.optim',fromlist='optim').__dict__[cfg.OPTIMIZER.opt_name](model.parameters(), lr=cfg.OPTIMIZER.lr, **cfg.OPTIMIZER.params)        
        if cfg.SCHEDULER.name is not None:                        
            scheduler = __import__('torch.optim.lr_scheduler', fromlist='lr_scheduler').__dict__[cfg.SCHEDULER.name](optimizer, **cfg.SCHEDULER.params)
        else:
            scheduler = None
            
        # Init Dataloader 
        trainloader, testloader = loader_dict[current_class_name]['train'],loader_dict[current_class_name]['test']
        
        model, trainloader, testloader, optimizer, scheduler = accelerator.prepare(model, trainloader, testloader, optimizer, scheduler)
        
        model = model.cuda()
        # Train 
        for epoch in range(epochs):
            # train one epoch 
            train(
                    model        = model, 
                    dataloader   = trainloader, 
                    testloader   = testloader, 
                    optimizer    = optimizer, 
                    scheduler    = scheduler,
                    accelerator  = accelerator, 
                    log_interval = log_interval,
                    epoch        = epoch,
                    epochs       = epochs,
                    savedir      = savedir, 
                    cfg          = cfg,
                    drift_monitor= drift_monitor 
                )
             
            if scheduler:
                scheduler.step()
                
                epoch_time_m.update(time.time() - end)
                end = time.time()
                
            if (epoch%2 == 0) or (epoch%199 == 0): 
                test_metrics = test(
                    model              = model, 
                    dataloader         = testloader, 
                    device             = accelerator.device, 
                    savedir            = savedir, 
                    use_wandb          = use_wandb,
                    epoch              = epoch, 
                    optimizer          = optimizer,
                    epoch_time_m       = epoch_time_m, 
                    class_name         = current_class_name,
                    current_class_name = current_class_name
                )
                    
        
        # EVALUATION
        num_current_class = list(loader_dict.keys()).index(current_class_name)
        
        # model save
        os.makedirs(f"{savedir}/model_weight/", exist_ok=True)
        torch.save(model.state_dict(),f"{savedir}/model_weight/{current_class_name}_model.pth")
    
        if cfg.CONTINUAL.continual:
            # Continual method 
            _logger.info('Continual Learning consolidate')
            # model.consolidate(trainloader)
            
            # Continual evaluation 
            num_start = 0 
            num_end = num_current_class+1 if num_current_class == len(loader_dict)-1 else num_current_class+2
        
            for n_task in range(num_start,num_end):
                print('\n')
                print(f"loader_dict : {len(list(loader_dict.items()))}")
                print(f"n_task : {n_task}")
                print('\n')
                class_name, class_loader_dict = list(loader_dict.items())[n_task]
                trainloader, testloader = loader_dict[class_name]['train'],loader_dict[class_name]['test']
                trainloader, testloader = accelerator.prepare(trainloader, testloader)            
                
                test_metrics = test(
                                model              = model, 
                                device             = accelerator.device,
                                savedir            = savedir, 
                                use_wandb          = use_wandb,
                                epoch              = 0 if epochs == 0 else epoch,
                                optimizer          = optimizer, 
                                epoch_time_m       = epoch_time_m,
                                class_name         = trainloader.dataset.class_name,
                                current_class_name = current_class_name,
                                dataloader         = testloader,
                                last               = True
                            )
        else:
            model  = __import__('models').__dict__[cfg.MODEL.method](
                backbone    = cfg.MODEL.backbone,
                **cfg.MODEL.params
                ).cuda()
            model = accelerator.prepare(model)
            _logger.info('Model init')