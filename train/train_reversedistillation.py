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
from utils.log import AverageMeter, metric_logging, DriftMonitor, save_performance_summary, generate_performance_report
import warnings
warnings.filterwarnings('ignore')

# Import for performance monitoring
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("thop not available. FLOPs calculation will be skipped.")

_logger = logging.getLogger('train')

# Performance monitoring utility class
class PerformanceMonitor:
    def __init__(self, device):
        self.device = device
        self.reset()
    
    def reset(self):
        self.total_samples = 0
        self.total_time = 0.0
        self.max_memory = 0.0
        self.flops_calculated = False
        self.flops = 0
        self.params = 0
        
    def start_batch(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.batch_start_time = time.time()
        
    def end_batch(self, batch_size):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_time = time.time() - self.batch_start_time
        self.total_time += batch_time
        self.total_samples += batch_size
        
        # Update GPU memory usage
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            self.max_memory = max(self.max_memory, current_memory)
    
    def calculate_flops(self, model, input_tensor):
        if THOP_AVAILABLE and not self.flops_calculated:
            try:
                model_copy = model
                flops, params = profile(model_copy, inputs=(input_tensor,), verbose=False)
                self.flops = flops
                self.params = params
                self.flops_calculated = True
            except Exception as e:
                _logger.warning(f"FLOPs calculation failed: {e}")
                self.flops = 0
                self.params = 0
    
    def get_metrics(self):
        throughput = self.total_samples / self.total_time if self.total_time > 0 else 0
        return {
            'throughput_samples_per_sec': throughput,
            'max_gpu_memory_gb': self.max_memory,
            'total_samples': self.total_samples,
            'total_time_sec': self.total_time,
            'flops': self.flops,
            'params': self.params
        }
    
    def log_metrics(self, prefix=""):
        metrics = self.get_metrics()
        _logger.info(f"{prefix} Performance Metrics:")
        _logger.info(f"  Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/sec")
        _logger.info(f"  Max GPU Memory: {metrics['max_gpu_memory_gb']:.3f} GB")
        _logger.info(f"  Total Samples: {metrics['total_samples']}")
        _logger.info(f"  Total Time: {metrics['total_time_sec']:.2f} sec")
        if THOP_AVAILABLE and metrics['flops'] > 0:
            flops_str, params_str = clever_format([metrics['flops'], metrics['params']], "%.3f")
            _logger.info(f"  FLOPs: {flops_str}")
            _logger.info(f"  Params: {params_str}")
        return metrics
    

def train(model, dataloader, testloader, optimizer, scheduler, accelerator, log_interval: int, epoch, epochs, savedir, cfg) -> dict:
    
    def collect_gradients(cfg, model, all_gradients, epoch, step):
        if ((cfg.CONTINUAL.online and (step % 10 == 0)) or (not cfg.CONTINUAL.online and ((epoch) % 2 == 0) and (step % 4 == 0))):
            step_grad_dict = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    step_grad_dict[name] = param.grad.clone().detach().cpu().numpy()
            all_gradients.append(step_grad_dict)
    
    def log_training_info(step, accelerator, dataloader, epoch, epochs, losses_m, optimizer, batch_time_m, data_time_m, images, perf_monitor):
        metrics = perf_monitor.get_metrics()
        _logger.info(
            'Train Epoch [{epoch}/{epochs}] " [{:d}/{}] Total Loss: {loss.avg:>6.4f} '
            'LR: {lr:.3e} '
            'Time: {batch_time.avg:.3f}s, {rate_avg:>3.2f}/s '
            'Data: {data_time.avg:.3f}s '
            'Throughput: {throughput:.2f} samples/s '
            'GPU Mem: {gpu_mem:.3f}GB'.format(
                (step + 1) // accelerator.gradient_accumulation_steps,
                len(dataloader) // accelerator.gradient_accumulation_steps,
                epoch=epoch, epochs=epochs, loss=losses_m,
                lr=optimizer.param_groups[0]['lr'], batch_time=batch_time_m,
                rate_avg=images[0].size(0) / batch_time_m.avg, data_time=data_time_m,
                throughput=metrics['throughput_samples_per_sec'],
                gpu_mem=metrics['max_gpu_memory_gb']
            )
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
            os.makedirs(f"{savedir}/gradients", exist_ok=True)
            np.save(f"{savedir}/gradients/{current_class_name_}_gradient_log_epoch_{epoch}.npy", all_gradients)
    
    # Initialize performance monitor
    perf_monitor = PerformanceMonitor(accelerator.device)
    
    batch_time_m = AverageMeter(); data_time_m = AverageMeter(); losses_m = AverageMeter()
    current_class_name = dataloader.dataset.class_name
    end = time.time()
    
    model.train()
    all_gradients = []
    for step, (images, _) in enumerate(dataloader):
        data_time_m.update(time.time() - end)
        
        # Start performance monitoring
        perf_monitor.start_batch()
        
        outputs = model(images) 
        
        # Calculate FLOPs on first batch
        if step == 0:
            perf_monitor.calculate_flops(model, images)
        
        loss = model.criterion(outputs)
        optimizer.zero_grad()
        accelerator.backward(loss)
        
        losses_m.update(loss.item())
        collect_gradients(cfg, model, all_gradients, epoch, step)
        optimizer.step()
        
        # End performance monitoring
        perf_monitor.end_batch(images.size(0))
        
        batch_time_m.update(time.time() - end)
        
        adjusted_log_interval = log_interval if cfg.CONTINUAL.online else 1
        if (step + 1) % adjusted_log_interval == 0:
            log_training_info(step, accelerator, dataloader, epoch, epochs, losses_m, optimizer, batch_time_m, data_time_m, images, perf_monitor)
        do_online_inference(cfg, step, dataloader, model, accelerator, savedir, epoch, testloader, current_class_name, batch_time_m, optimizer)
        end = time.time()
    
    # Log final training performance metrics
    train_metrics = perf_monitor.log_metrics("Training")
    
    # Save performance metrics
    os.makedirs(f"{savedir}/performance_logs", exist_ok=True)
    with open(f"{savedir}/performance_logs/{current_class_name}_train_epoch_{epoch}_performance.txt", "w") as f:
        for key, value in train_metrics.items():
            f.write(f"{key}: {value}\n")
    
    save_gradients_if_needed(cfg, dataloader, epoch, savedir, all_gradients)
    return {
        "loss": losses_m.avg, 
        "gradients": all_gradients, 
        "class_name": current_class_name,
        "performance_metrics": train_metrics
    }



def test(model, dataloader, device, 
         savedir, use_wandb, epoch, optimizer, epoch_time_m, class_name, current_class_name,
         last : bool = False) -> dict:
    from utils.metrics import MetricCalculator, loco_auroc   
    
    # Initialize performance monitor for inference
    perf_monitor = PerformanceMonitor(device)
    
    model.eval()
    img_level = MetricCalculator(metric_list = ['auroc','average_precision'])
    pix_level = MetricCalculator(metric_list = ['auroc','average_precision'])     

    #! Inference     
    for idx, (images, labels, gts) in enumerate(dataloader):
        
        # Start performance monitoring
        perf_monitor.start_batch()

        with torch.no_grad():
            outputs = model(images)   
            score_map = model.get_score_map(outputs).detach().cpu()
            score = score_map.reshape(score_map.shape[0],-1).max(-1)[0]
        
        # Calculate FLOPs on first batch
        if idx == 0:
            perf_monitor.calculate_flops(model, images)
        
        # End performance monitoring
        perf_monitor.end_batch(images.size(0))
                
        # Stack Scoring for metrics 
        pix_level.update(score_map,gts.type(torch.int))
        img_level.update(score, labels.type(torch.int))
    
    # Log inference performance metrics
    inference_metrics = perf_monitor.log_metrics("Inference")
            
    i_results, p_results = img_level.compute(), pix_level.compute()
    _logger.info(f"Current Class name : {current_class_name} Class name : {class_name} Image AUROC: {i_results['auroc']:.3f}| Pixel AUROC: {p_results['auroc']:.3f}")
        
    test_result = OrderedDict(img_level = i_results)
    test_result.update([('pix_level', p_results)])
    test_result.update([('inference_performance', inference_metrics)])
    
    # Save inference performance metrics
    os.makedirs(f"{savedir}/performance_logs", exist_ok=True)
    with open(f"{savedir}/performance_logs/{class_name}_test_epoch_{epoch}_performance.txt", "w") as f:
        for key, value in inference_metrics.items():
            f.write(f"{key}: {value}\n")
    
    metric_logging(
            savedir = savedir, use_wandb = use_wandb, epoch = epoch,
            optimizer = optimizer, epoch_time_m = epoch_time_m,
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
        
        # Train 
        for epoch in range(epochs):
            # train one epoch 
            train_result = train(
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
                    cfg          = cfg 
                )                
            if scheduler:
                scheduler.step()
                
                epoch_time_m.update(time.time() - end)
                end = time.time()
                    
        
        # EVALUATION
        num_current_class = list(loader_dict.keys()).index(current_class_name)
        
        # model save
        os.makedirs(f"{savedir}/model_weight/", exist_ok=True)
        torch.save(model.state_dict(),f"{savedir}/model_weight/{current_class_name}_model.pth")
    
        if cfg.CONTINUAL.continual:
            # Continual method 
            _logger.info('Continual Learning consolidate')
            model.consolidate(trainloader)
            
            # Continual evaluation 
            num_start = 0 
            num_end = num_current_class+2
        
            for n_task in range(num_start,num_end):
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
                
                # Save performance summary for continual learning
                if 'performance_metrics' in train_result and 'inference_performance' in test_metrics:
                    save_performance_summary(
                        savedir=savedir,
                        class_name=class_name,
                        train_metrics=train_result['performance_metrics'],
                        test_metrics=test_metrics['inference_performance'],
                        epoch=epoch
                    )
        else:
            # For non-continual learning, also run test to get inference performance
            test_metrics = test(
                model=model,
                device=accelerator.device,
                savedir=savedir,
                use_wandb=use_wandb,
                epoch=0 if epochs == 0 else epoch-1,  # Use last epoch
                optimizer=optimizer,
                epoch_time_m=epoch_time_m,
                class_name=current_class_name,
                current_class_name=current_class_name,
                dataloader=testloader,
                last=True
            )
            
            # Save performance summary for non-continual learning
            if 'performance_metrics' in train_result and 'inference_performance' in test_metrics:
                save_performance_summary(
                    savedir=savedir,
                    class_name=current_class_name,
                    train_metrics=train_result['performance_metrics'],
                    test_metrics=test_metrics['inference_performance'],
                    epoch=epoch-1 if epoch > 0 else 0
                )
            
            model  = __import__('models').__dict__[cfg.MODEL.method](
                backbone    = cfg.MODEL.backbone,
                **cfg.MODEL.params
                )
            model = accelerator.prepare(model)
            _logger.info('Model init')
    
    # Generate final performance report
    _logger.info("Generating comprehensive performance report...")
    generate_performance_report(savedir)