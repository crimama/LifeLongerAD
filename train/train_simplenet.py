import wandb 
import logging
import time
import os 
import numpy as np
import pandas as pd

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from datasets.mvtecad import class_label_mapping
from collections import OrderedDict
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from utils.metrics import MetricCalculator
from utils.log import AverageMeter,metric_logging,DriftMonitor
import warnings
warnings.filterwarnings('ignore')

_logger = logging.getLogger('train')
    

def train(model, dataloader, optimizer, accelerator, log_interval: int, epoch, epochs, cfg) -> dict:
    
    def collect_gradients(cfg, model, all_gradients, epoch, step):
        if ((cfg.CONTINUAL.online and (step % 10 == 0)) or (not cfg.CONTINUAL.online and ((epoch) % 2 == 0) and (step % 4 == 0))):
            step_grad_dict = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    step_grad_dict[name] = param.grad.clone().detach().cpu().numpy()
            all_gradients.append(step_grad_dict)
    
    def log_training_info(step, accelerator, dataloader, epoch, epochs,
                      losses_m,
                      optimizer, batch_time_m, data_time_m, images, wandb_use:bool = False):
        current_step = (step + 1) // accelerator.gradient_accumulation_steps
        total_steps = len(dataloader) // accelerator.gradient_accumulation_steps
        _logger.info(
            'Train Epoch [{epoch}/{epochs}] [{current_step:d}/{total_steps:d}] '
            'Total Loss: {loss_val:>6.4f} | '
            'LR: {lr:.3e} | '
            'Time: {batch_time_avg:.3f}s, {rate_avg:>3.2f}/s | '
            'Data: {data_time_avg:.3f}s'.format(
                current_step=current_step,
                total_steps=total_steps,
                epoch=epoch,
                epochs=epochs,
                loss_val=losses_m.val,
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
                'Train/Learning Rate': optimizer['dsc_opt'].param_groups[0]['lr'],
                'Time/Train Batch Average (s)': batch_time_m.avg,
                'Time/Processing Rate (img/s)': images[0].size(0) / batch_time_m.avg,
                'Time/Data Loading Average (s)': data_time_m.avg,
                'Train/Total Loss (val)': losses_m.val,
            }
        )    
    
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    
    losses_m = AverageMeter()
    
    current_class_name = dataloader.dataset.class_name
    model.train()
    all_gradients = []
    
    end = time.time()
    dsc_opt = optimizer['dsc_opt']
    pre_projection_opt = optimizer['proj_opt']
    
    for step, (images, labels, class_labels) in enumerate(dataloader):        
        data_time_m.update(time.time() - end)    
        
        loss = model.train_discriminator(images)         
        
        dsc_opt.zero_grad()
        pre_projection_opt.zero_grad()
        accelerator.backward(loss)  
        dsc_opt.step()
        pre_projection_opt.step()
        
        # Loss record  
        losses_m.update(loss.item())
        
        batch_time_m.update(time.time() - end)        
        # Logging 
        adjusted_log_interval = log_interval if cfg.CONTINUAL.online else 1
        if (step + 1) % adjusted_log_interval == 0:
            log_training_info(step, accelerator, dataloader, epoch, epochs, 
                            losses_m ,
                            optimizer['dsc_opt'], batch_time_m, data_time_m, images, wandb_use=cfg.TRAIN.wandb.use)
                    
        end = time.time()
    
    return {"loss": losses_m.avg, "gradients": all_gradients, "class_name": current_class_name}

def test(model, dataloader,
         savedir, use_wandb, epoch, optimizer, class_name, current_class_name,
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
            score, score_map = model.predict(images)   
                
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
            optimizer = optimizer['dsc_opt'], epoch_time_m = test_time_m,   
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
    
    # Set for continual learning 
    model.train()
    model = model.cuda()
        
    ## 희소성 설정 정의
    
    epoch_time_m = AverageMeter()
    end = time.time() 
    
    for n_task, (current_class_name, class_loader_dict) in enumerate(loader_dict.items()):        
        
        best_score = 0.0
        if (n_task == 0) or (cfg.CONTINUAL.continual==False):
            drift_monitor = DriftMonitor(log_dir=os.path.join(savedir,'DriftMonitor.log'))
        
        torch.cuda.empty_cache()
        _logger.info(f"Current Class Name : {current_class_name}")        
            
        # Init optimzier & SCheduler 
        dsc_opt = model.dsc_opt
        pre_projection_opt = model.proj_opt                
            
        # Init Dataloader 
        trainloader, testloader = loader_dict[current_class_name]['train'],loader_dict[current_class_name]['test']
        
        model, trainloader, testloader, dsc_opt, pre_projection_opt = accelerator.prepare(model, trainloader, testloader, dsc_opt, pre_projection_opt)
        
        model = model.cuda()
        # Train 
        for epoch in range(epochs):
            # train one epoch 
            train(
                    model        = model, 
                    dataloader   = trainloader, 
                    optimizer    = {'dsc_opt': dsc_opt, 'proj_opt': pre_projection_opt},
                    accelerator  = accelerator, 
                    log_interval = log_interval,
                    epoch        = epoch,
                    epochs       = epochs,
                    cfg          = cfg,
                )
                
            epoch_time_m.update(time.time() - end)
            end = time.time()                
                
            if (epoch%10 == 0) or (epoch%199 == 0): 
                test_metrics = test(
                    model              = model, 
                    dataloader         = testloader, 
                    savedir            = savedir, 
                    use_wandb          = use_wandb,
                    epoch              = epoch, 
                    optimizer          = {'dsc_opt': dsc_opt, 'proj_opt': pre_projection_opt},
                    class_name         = current_class_name,
                    current_class_name = current_class_name
                )
                    
        
            # EVALUATION
            num_current_class = list(loader_dict.keys()).index(current_class_name)            
            # model save
            score = (test_metrics['img_level']['auroc'] + test_metrics['pix_level']['auroc']) / 2
            if best_score < score:
                os.makedirs(f"{savedir}/model_weight/", exist_ok=True)
                torch.save(model.state_dict(),f"{savedir}/model_weight/{current_class_name}_model.pth")
                best_score = score 
    
        if cfg.CONTINUAL.continual:
            # Continual method 
            _logger.info('Continual Learning consolidate')                        
            
            # Continual evaluation 
            num_start = 0 
            num_end = num_current_class+1 if num_current_class == len(loader_dict)-1 else num_current_class+2
        
            for n_task in range(num_start,num_end):
                _logger.info(f"Evaluating task {n_task+1}/{len(loader_dict)}")
                class_name, class_loader_dict = list(loader_dict.items())[n_task]
                trainloader, testloader = loader_dict[class_name]['train'],loader_dict[class_name]['test']
                trainloader, testloader = accelerator.prepare(trainloader, testloader)            
                
                test_metrics = test(
                                model              = model, 
                                savedir            = savedir, 
                                use_wandb          = use_wandb,
                                epoch              = 0 if epochs == 0 else epoch,
                                optimizer          = {'dsc_opt': dsc_opt, 'proj_opt': pre_projection_opt},
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