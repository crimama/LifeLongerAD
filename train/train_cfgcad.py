import wandb 
import logging
import time
import os 
import numpy as np
import pandas as pd
import signal
import sys
import gc
import psutil
import omegaconf

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from datasets.mvtecad import class_label_mapping
from collections import OrderedDict
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from utils.metrics import MetricCalculator
from utils.log import AverageMeter,metric_logging,DriftMonitor
from CL import CL_Transformer
import warnings
warnings.filterwarnings('ignore')

_logger = logging.getLogger('train')

# Global flag for graceful shutdown
_shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    _logger.info(f"Received signal {signum}. Requesting graceful shutdown...")

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def check_system_resources():
    """Monitor system resources and warn if running low."""
    try:
        # Memory check
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            _logger.warning(f"High memory usage: {memory.percent:.1f}% used")
            
        # Disk space check
        disk = psutil.disk_usage('/')
        if disk.percent > 90:
            _logger.warning(f"Low disk space: {disk.percent:.1f}% used")
            
        # GPU memory check if CUDA is available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.memory_allocated(i) / torch.cuda.max_memory_allocated(i) * 100
                if gpu_memory > 90:
                    _logger.warning(f"High GPU {i} memory usage: {gpu_memory:.1f}%")
                    
    except Exception as e:
        _logger.debug(f"Resource monitoring error: {e}")

def safe_save_checkpoint(model, optimizer, epoch, class_name, savedir, best_score):
    """Safely save model checkpoint with error handling."""
    try:
        os.makedirs(f"{savedir}/model_weight/", exist_ok=True)
        checkpoint_path = f"{savedir}/model_weight/{class_name}_model.pth"
        temp_path = checkpoint_path + ".tmp"
        
        # Save to temporary file first
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_score': best_score,
            'class_name': class_name
        }, temp_path)
        
        # Atomic move
        os.rename(temp_path, checkpoint_path)
        _logger.info(f"Checkpoint saved: {checkpoint_path}")
        return True
        
    except Exception as e:
        _logger.error(f"Failed to save checkpoint: {e}")
        return False

def cleanup_gpu_memory():
    """Clean up GPU memory."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    except Exception as e:
        _logger.debug(f"GPU cleanup error: {e}")

def safe_wandb_log(metrics, retry_count=3):
    """Safely log to wandb with retry mechanism."""
    if not wandb.run:
        return
        
    for attempt in range(retry_count):
        try:
            wandb.log(metrics)
            return
        except Exception as e:
            _logger.warning(f"wandb log attempt {attempt + 1} failed: {e}")
            if attempt < retry_count - 1:
                time.sleep(1)  # Brief pause before retry
            else:
                _logger.error("Failed to log to wandb after all retries")

def train(model, dataloader, testloader, optimizer, scheduler, accelerator, log_interval: int, epoch, epochs, savedir, cfg, drift_monitor, cl_manager) -> dict:
    
    def collect_gradients(cfg, model, all_gradients, epoch, step):
        try:
            if ((cfg.CONTINUAL.online and (step % 10 == 0)) or (not cfg.CONTINUAL.online and ((epoch) % 2 == 0) and (step % 4 == 0))):
                step_grad_dict = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        step_grad_dict[name] = param.grad.clone().detach().cpu().numpy()
                all_gradients.append(step_grad_dict)
        except Exception as e:
            _logger.warning(f"Failed to collect gradients: {e}")
    
    def log_training_info(step, accelerator, dataloader, epoch, epochs,
                      losses_m, feature_losses_m, svd_losses_m,
                      optimizer, batch_time_m, data_time_m, images, wandb_use:bool = False):
        try:
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
                metrics = {
                    'Train/Epoch': epoch,
                    'Train/Total Loss': losses_m.avg,
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
                safe_wandb_log(metrics)
        except Exception as e:
            _logger.error(f"Logging failed: {e}")

    
    def do_online_inference(cfg, step, dataloader, model, accelerator, savedir, epoch, testloader, current_class_name, batch_time_m, optimizer, cl_manager):
        try:
            if ((cfg.CONTINUAL.online and (step % 10 == 0)) or (step == len(dataloader) - 1)):
                test_metrics = test(
                    model=model, device=accelerator.device, savedir=savedir, use_wandb=False,
                    epoch=step if cfg.CONTINUAL.online else step*epoch, optimizer=optimizer,
                    epoch_time_m=batch_time_m, class_name=current_class_name,
                    current_class_name=current_class_name, dataloader=testloader, cl_manager=cl_manager
                )
        except Exception as e:
            _logger.warning(f"Online inference failed: {e}")
    
    def save_gradients_if_needed(cfg, dataloader, epoch, savedir, all_gradients):
        try:
            if (cfg.CONTINUAL.online or (not cfg.CONTINUAL.online and ((epoch) % 2 == 0))):
                current_class_name_ = dataloader.dataset.class_name
                gradient_dir = f"{savedir}/gradients"
                os.makedirs(gradient_dir, exist_ok=True)
                np.save(f"{gradient_dir}/{current_class_name_}_gradient_log_epoch_{epoch}.npy", all_gradients)
        except Exception as e:
            _logger.warning(f"Failed to save gradients: {e}")
    
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
        global _shutdown_requested
        if _shutdown_requested:
            _logger.info("Shutdown requested during training. Stopping gracefully...")
            break
            
        try:
            Input = {'image':images,'clslabel':class_labels}
            data_time_m.update(time.time() - end)
            
            # PGPT: Inject prompt if enabled
            if cl_manager.use_pgpt and cl_manager.current_class_name:
                prompts = cl_manager.get_prompts_for_class(cl_manager.current_class_name)
                if prompts:
                    # Use the first prompt for simplicity
                    prompt = prompts[0]
                    Input = cl_manager.inject_prompt_into_model(prompt, Input)
                    # Reduce debug prints during training - only print occasionally
                    if step % 50 == 0:  # Only print every 50 steps
                        print(f"✓ PGPT: Using prompt for class '{cl_manager.current_class_name}'")
            
            # Training with Enhanced Continual Learning
            cl_manager.save_old_tasks_weights() # 가중치 저장
            
            outputs = model(Input) 
            
            # Calculate loss
            loss = model.criterion(outputs, Input, skip=False, cl_manager=cl_manager)
            
            optimizer.zero_grad()
            accelerator.backward(loss['loss'])         
            
            # Loss record with enhanced metrics
            losses_m.update(loss['loss'].item())
            feature_losses_m.update(loss['feature_loss'])
            svd_losses_m.update(loss['svd_loss'])
            
            optimizer.step()
            cl_manager.recover_old_tasks_weights() # 이전 작업 가중치 복구
            
            batch_time_m.update(time.time() - end)        
            # Enhanced Logging 
            adjusted_log_interval = log_interval if cfg.CONTINUAL.online else 1
            if (step + 1) % adjusted_log_interval == 0:
                log_training_info(step, accelerator, dataloader, epoch, epochs, 
                                losses_m, feature_losses_m, svd_losses_m,
                                optimizer, batch_time_m, data_time_m, images, wandb_use=cfg.TRAIN.wandb.use)
                
                # Check resources periodically
                if (step + 1) % (adjusted_log_interval * 5) == 0:
                    check_system_resources()
                
            do_online_inference(cfg, step, dataloader, model, accelerator, savedir, epoch, testloader, current_class_name, batch_time_m, optimizer, cl_manager)
            end = time.time()
            
        except Exception as e:
            _logger.error(f"Training step {step} failed: {e}")
            # Clean up and continue
            cleanup_gpu_memory()
            continue
    
    return {"loss": losses_m.avg, "gradients": all_gradients, "class_name": current_class_name}

def test(model, dataloader, device, 
         savedir, use_wandb, epoch, optimizer, epoch_time_m, class_name, current_class_name,
         cl_manager=None, last : bool = False) -> dict:
    try:
        from utils.metrics import MetricCalculator, loco_auroc    
        model.eval()
        img_level = MetricCalculator(metric_list = ['auroc','average_precision'])
        pix_level = MetricCalculator(metric_list = ['auroc','average_precision'])     

        test_time_m = AverageMeter()    
        
        #! Inference     
        end = time.time()
        for idx, (images, labels, class_labels, gts) in enumerate(dataloader):
            global _shutdown_requested
            if _shutdown_requested:
                _logger.info("Shutdown requested during testing. Stopping gracefully...")
                break

            with torch.no_grad():
                Input = {'image':images,'clsname':class_labels}
                
                # PGPT: Select prompt using K-NN during inference
                if cl_manager is not None and cl_manager.use_pgpt:
                    # Extract features for prompt selection
                    if hasattr(model, 'backbone'):
                        features = model.backbone(images)
                    else:
                        # Fallback: use the model to get features
                        temp_output = model(Input)
                        features = temp_output.get('feature_align', temp_output)
                    
                    # Average pooling if needed
                    if features.dim() > 2:
                        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
                    
                    # Select best prompt using K-NN
                    prompt, selected_class = cl_manager.select_prompt_by_knn(features)
                    if prompt is not None:
                        Input = cl_manager.inject_prompt_into_model(prompt, Input)
                        # Reduce debug prints during testing
                        if idx % 10 == 0:  # Only print every 10th batch
                            print(f"✓ PGPT: Selected prompt for class '{selected_class}' during inference")
                
                outputs = model(Input)   
                score_map = outputs['pred'].detach().cpu()            
                score = score_map.reshape(score_map.shape[0],-1).max(-1)[0]
                    
            # Stack Scoring for metrics 
            pix_level.update(score_map,gts.type(torch.int))
            img_level.update(score, labels.type(torch.int))
            
            test_time_m.update(time.time() - end)
            end = time.time()
                
        i_results, p_results = img_level.compute(), pix_level.compute()
        _logger.info("=" * 60)
        _logger.info(f"METRICS - Current Class: {current_class_name} | Target Class: {class_name}")
        _logger.info(f"Image AUROC: {i_results['auroc']:.3f} | Pixel AUROC: {p_results['auroc']:.3f}")
        
        # PGPT: Add PGPT status to metric logging if enabled
        if cl_manager is not None and cl_manager.use_pgpt:
            _logger.info(f"PGPT Status: {len(cl_manager.prompt_pool)} classes with prompts, {len(cl_manager.prototype_repository)} prototypes calculated")
        
        _logger.info("=" * 60)
            
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
        
    except Exception as e:
        _logger.error(f"Testing failed: {e}")
        # Return dummy results to prevent crash
        return OrderedDict(img_level={'auroc': 0.0}, pix_level={'auroc': 0.0})


def fit(
    model, loader_dict:dict, accelerator,
    epochs: int, use_wandb: bool, log_interval: int, eval_interval: int, seed: int = None, savedir: str = None
    ,cfg=None):
    try:
        print(savedir)    
        
        # Set up logging for container environment
        _logger.info(f"Starting training in container. PID: {os.getpid()}")
        _logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Set for continual learning 
        model.train()
        model = model.cuda()
        task_labels = [] 
        for k,v in loader_dict.items():
            # Handle case where k might be a list of class names or a single class name
            print(f"Processing loader_dict key: {k} (type: {type(k)})")
            try:
                if isinstance(k, (list, tuple)) or isinstance(k, omegaconf.listconfig.ListConfig):
                    # If k is a list/tuple/ListConfig of class names, get mapping for each
                    print(f"Key is a list/tuple/ListConfig, processing each item: {k}")
                    temp = []
                    for class_name in k:
                        if class_name not in class_label_mapping:
                            _logger.error(f"Class name '{class_name}' not found in class_label_mapping")
                            _logger.error(f"Available keys: {list(class_label_mapping.keys())}")
                            raise KeyError(f"Class name '{class_name}' not found in class_label_mapping")
                        temp.append(class_label_mapping[class_name])
                else:
                    # If k is a single class name string
                    print(f"Key is a single class name: {k}")
                    if k not in class_label_mapping:
                        _logger.error(f"Class name '{k}' not found in class_label_mapping")
                        _logger.error(f"Available keys: {list(class_label_mapping.keys())}")
                        raise KeyError(f"Class name '{k}' not found in class_label_mapping")
                    temp = class_label_mapping[k]
                task_labels.append(temp)
                print(f"Successfully mapped {k} to {temp}")
            except Exception as e:
                _logger.error(f"Error processing key {k}: {e}")
                raise
            
        ## Enhanced Continual Learning Configuration
        
        # Initialize enhanced CL manager with new features
        cl_manager = CL_Transformer(
            model=model, 
            device=accelerator.device, 
            use_pgpt=cfg.CONTINUAL.get('use_pgpt', False),
            prompt_dim=cfg.CONTINUAL.get('prompt_dim', 256),
            num_prompts_per_class=cfg.CONTINUAL.get('num_prompts_per_class', 1),
            k_neighbors=cfg.CONTINUAL.get('k_neighbors', 1)
        )
        
        cl_manager.set_init_network_weight()
        
        epoch_time_m = AverageMeter()
        end = time.time()

        optimizer = __import__('torch.optim',fromlist='optim').__dict__[cfg.OPTIMIZER.opt_name](model.parameters(), lr=cfg.OPTIMIZER.lr, **cfg.OPTIMIZER.params)        
        if cfg.SCHEDULER.name is not None:                        
            scheduler = __import__('torch.optim.lr_scheduler', fromlist='lr_scheduler').__dict__[cfg.SCHEDULER.name](optimizer, **cfg.SCHEDULER.params)
        else:
            scheduler = None
        
        for n_task, (current_class_name, class_loader_dict) in enumerate(loader_dict.items()):
            global _shutdown_requested
            if _shutdown_requested:
                _logger.info("Shutdown requested. Exiting task loop...")
                break
            
            # Enhanced SCL with knowledge preservation
            cl_manager.reset_importance() # 새 작업 시작 시 중요도 리셋
            
            # PGPT: Initialize prompts for new class
            if cl_manager.use_pgpt:
                cl_manager.initialize_prompt_for_class(current_class_name)
                cl_manager.set_current_class(current_class_name)
                _logger.info(f"PGPT: Initialized prompts for class '{current_class_name}'")
            
            best_score = 0.0
            if (n_task == 0) or (cfg.CONTINUAL.continual==False):
                drift_monitor = DriftMonitor(log_dir=os.path.join(savedir,'DriftMonitor.log'))
            
            cleanup_gpu_memory()
            _logger.info(f"Current Class Name : {current_class_name}")        
            _logger.info(f"Enhanced CL Features: PGPT={cl_manager.use_pgpt}")
                
            # Init optimzier & SCheduler         
            # Init Dataloader 
            trainloader, testloader = loader_dict[current_class_name]['train'],loader_dict[current_class_name]['test']
            
            model, trainloader, testloader, optimizer, scheduler = accelerator.prepare(model, trainloader, testloader, optimizer, scheduler)
            
            model = model.cuda()
            # Enhanced Training Loop
            for epoch in range(epochs):
                if _shutdown_requested:
                    _logger.info("Shutdown requested. Exiting epoch loop...")
                    break
                    
                try:
                    # train one epoch with enhanced features
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
                            cfg          = cfg,
                            drift_monitor= drift_monitor,
                            cl_manager   = cl_manager
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
                            current_class_name = current_class_name,
                            cl_manager         = cl_manager
                        )
                                
                
                    # EVALUATION
                    num_current_class = list(loader_dict.keys()).index(current_class_name)            
                    # model save
                    score = (test_metrics['img_level']['auroc'] + test_metrics['pix_level']['auroc']) / 2
                    if best_score < score:
                        safe_save_checkpoint(model, optimizer, epoch, current_class_name, savedir, score)
                        best_score = score 
                        
                except Exception as e:
                    _logger.error(f"Epoch {epoch} failed: {e}")
                    cleanup_gpu_memory()
                    continue
        
            if cfg.CONTINUAL.continual:
                try:
                    
                    # Enhanced Continual evaluation with detailed logging
                    num_start = 0 
                    num_end = num_current_class+1 if num_current_class == len(loader_dict)-1 else num_current_class+2
                
                    for n_task in range(num_start,num_end):
                        if _shutdown_requested:
                            break
                            
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
                                        epoch              = epochs - 1 if epochs > 0 else 0,  # Use final epoch number
                                        optimizer          = optimizer, 
                                        epoch_time_m       = epoch_time_m,
                                        class_name         = trainloader.dataset.class_name,
                                        current_class_name = current_class_name,
                                        cl_manager         = cl_manager,
                                        last               = True
                                    )
                    if n_task < len(loader_dict) - 1:
                        cl_manager.prepare_next_task()  # Enhanced preparation for next task
                        
                except Exception as e:
                    _logger.error(f"Continual learning evaluation failed: {e}")
            else:
                # Reinitialize model for non-continual learning
                try:
                    model  = __import__('models').__dict__[cfg.MODEL.method](
                        backbone    = cfg.MODEL.backbone,
                        **cfg.MODEL.params
                        ).cuda()
                    model = accelerator.prepare(model)
                    _logger.info('Model init')
                except Exception as e:
                    _logger.error(f"Model reinitialization failed: {e}")
            
            # PGPT: Calculate prototype for the current class after training
            if cl_manager.use_pgpt:
                try:
                    cl_manager.calculate_prototype(trainloader, current_class_name)
                    _logger.info(f"PGPT: Prototype calculated for class '{current_class_name}'")
                    
                    # Add PGPT summary at the end of each task
                    _logger.info("=" * 40)
                    _logger.info("PGPT SUMMARY")
                    _logger.info(f"Total classes with prompts: {len(cl_manager.prompt_pool)}")
                    _logger.info(f"Total prototypes calculated: {len(cl_manager.prototype_repository)}")
                    _logger.info(f"Current task: {n_task + 1}/{len(loader_dict)}")
                    _logger.info("=" * 40)
                    
                except Exception as e:
                    _logger.error(f"PGPT prototype calculation failed for '{current_class_name}': {e}")
                    
    except Exception as e:
        _logger.error(f"Training failed with error: {e}")
        raise
    finally:
        # Cleanup
        cleanup_gpu_memory()
        if wandb.run:
            try:
                wandb.finish()
            except:
                pass
        _logger.info("Training completed or terminated")