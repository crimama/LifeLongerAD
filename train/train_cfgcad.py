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
from CL import CL_Transformer
import warnings
warnings.filterwarnings('ignore')

_logger = logging.getLogger('train')
    

def train(model, dataloader, testloader, optimizer, scheduler, accelerator, log_interval: int, epoch, epochs, savedir, cfg, drift_monitor, cl_manager) -> dict:
    
    def collect_gradients(cfg, model, all_gradients, epoch, step):
        if ((cfg.CONTINUAL.online and (step % 10 == 0)) or (not cfg.CONTINUAL.online and ((epoch) % 2 == 0) and (step % 4 == 0))):
            step_grad_dict = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    step_grad_dict[name] = param.grad.clone().detach().cpu().numpy()
            all_gradients.append(step_grad_dict)
    
    def log_training_info(step, accelerator, dataloader, epoch, epochs,
                      losses_m, feature_losses_m, svd_losses_m, ce_losses_m,
                      optimizer, batch_time_m, data_time_m, images, wandb_use:bool = False):
        current_step = (step + 1) // accelerator.gradient_accumulation_steps
        total_steps = len(dataloader) // accelerator.gradient_accumulation_steps
        _logger.info(
            'Train Epoch [{epoch}/{epochs}] [{current_step:d}/{total_steps:d}] '
            'Total Loss: {loss_val:>6.4f} | '
            'Feature Loss: {feature_loss_val:>6.4f} | '            
            'SVD Loss: {svd_loss_val:>6.4f} | '
            'CE Loss: {ce_loss_val:>6.4f} | '
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
                ce_loss_val=ce_losses_m.val,
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
                'Train/CE Loss (val)': ce_losses_m.val,
            }
        )

    
    def do_online_inference(cfg, step, dataloader, model, accelerator, savedir, epoch, testloader, current_class_name, batch_time_m, optimizer):
        if ((cfg.CONTINUAL.online and (step % 10 == 0)) or (step == len(dataloader) - 1)):
            # Get metric_list from configuration if available
            metric_list = cfg.DATASET.get('metric_list', None) if hasattr(cfg.DATASET, 'metric_list') else None
            
            test_metrics = test(
                model=model, device=accelerator.device, savedir=savedir, use_wandb=False,
                epoch=step if cfg.CONTINUAL.online else step*epoch, optimizer=optimizer,
                epoch_time_m=batch_time_m, class_name=current_class_name,
                current_class_name=current_class_name, dataloader=testloader,
                metric_list=metric_list
            )
    
    def save_gradients_if_needed(cfg, dataloader, epoch, savedir, all_gradients):
        if (cfg.CONTINUAL.online or (not cfg.CONTINUAL.online and ((epoch) % 2 == 0))):
            current_class_name_ = dataloader.dataset.class_name
            np.save(f"{savedir}/gradients/{current_class_name_}_gradient_log_epoch_{epoch}.npy", all_gradients)
    
    # Helper function to safely call methods on CL manager
    def safe_cl_call(cl_manager, method_name, fallback_value=None):
        if cl_manager is None:
            return fallback_value
            
        if not hasattr(cl_manager, method_name):
            _logger.warning(f"CL manager missing method: {method_name}")
            return fallback_value
            
        try:
            method = getattr(cl_manager, method_name)
            return method()
        except Exception as e:
            _logger.error(f"Error calling {method_name}: {e}")
            return fallback_value
    
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    
    losses_m = AverageMeter()
    feature_losses_m = AverageMeter()
    svd_losses_m = AverageMeter()
    ce_losses_m = AverageMeter()

    current_class_name = dataloader.dataset.class_name
    model.train()
    all_gradients = []
    
    end = time.time()
    for step, (images, labels, class_labels) in enumerate(dataloader):
        Input = {'image':images,'clslabel':class_labels}
        data_time_m.update(time.time() - end)
        
        # Training 
        if cl_manager is not None:
            safe_cl_call(cl_manager, 'save_old_tasks_weights')
        
        outputs = model(Input) 
        loss = model.criterion(outputs, Input)        
        optimizer.zero_grad()
        accelerator.backward(loss['loss'])         
        
        # Loss record  
        losses_m.update(loss['loss'].item())
        feature_losses_m.update(loss['feature_loss'])
        svd_losses_m.update(loss['svd_loss'])
        ce_losses_m.update(loss['cls_loss'])
        
        if cl_manager is not None:
            safe_cl_call(cl_manager, 'apply_mask_on_grad')
        
        optimizer.step()
        
        if cl_manager is not None:
            safe_cl_call(cl_manager, 'calculate_importance')
            safe_cl_call(cl_manager, 'recover_old_tasks_weights')
        
        batch_time_m.update(time.time() - end)        
        # Logging 
        adjusted_log_interval = log_interval if cfg.CONTINUAL.online else 1
        if (step + 1) % adjusted_log_interval == 0:
            log_training_info(step, accelerator, dataloader, epoch, epochs, 
                            losses_m, feature_losses_m, svd_losses_m, ce_losses_m,
                            optimizer, batch_time_m, data_time_m, images, wandb_use=cfg.TRAIN.wandb.use)
            
        # do_online_inference(cfg, step, dataloader, model, accelerator, savedir, epoch, testloader, current_class_name, batch_time_m, optimizer)
        end = time.time()
    
    return {"loss": losses_m.avg, "gradients": all_gradients, "class_name": current_class_name}

def test(model, dataloader, device, 
         savedir, use_wandb, epoch, optimizer, epoch_time_m, class_name, current_class_name,
         last : bool = False, metric_list=None) -> dict:
    from utils.metrics import MetricCalculator, loco_auroc    
    model.eval()
    
    # 기본 메트릭 목록 정의
    default_img_metrics = ['auroc', 'average_precision']
    default_pix_metrics = ['auroc', 'average_precision']
    
    # cfg 에서 metric_list 파라미터가 제공된 경우 사용, 아니면 기본값 사용
    if metric_list is None:
        img_metrics = default_img_metrics
        pix_metrics = default_pix_metrics
    elif isinstance(metric_list, dict):
        img_metrics = metric_list.get('img_level', default_img_metrics)
        pix_metrics = metric_list.get('pix_level', default_pix_metrics)
    else:
        # 문자열 목록인 경우 이미지와 픽셀 레벨 모두에 동일하게 적용
        img_metrics = metric_list
        pix_metrics = metric_list + ['aupro'] if 'aupro' not in metric_list else metric_list
    
    img_level = MetricCalculator(metric_list=img_metrics)
    pix_level = MetricCalculator(metric_list=pix_metrics)     

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
    
    # 계산된 메트릭 결과 가져오기
    i_results, p_results = img_level.compute(), pix_level.compute()
    
    # 로그 메시지 동적 생성
    log_parts = [
        f"Current Class name : {current_class_name} Class name : {class_name}"
    ]
    
    # 이미지 레벨 메트릭 로깅
    for metric in img_metrics:
        if metric in i_results:
            log_parts.append(f"Image {metric}: {i_results[metric]:.3f}")
    
    # 픽셀 레벨 메트릭 로깅
    for metric in pix_metrics:
        if metric in p_results:
            log_parts.append(f"Pixel {metric}: {p_results[metric]:.3f}")
    
    # 로그 메시지 출력
    _logger.info(" | ".join(log_parts))
            
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
    
    # Set for continual learning 
    model.train()
    model = model.cuda()
    task_labels = [] 
    for k,v in loader_dict.items():
        temp = [class_label_mapping[k_] for k_ in k]
        task_labels.append(temp)
        
    ## Define sparsity configuration
    sparsity_config = cfg.CONTINUAL.method.params        
    
    # Determine whether to apply CL (only for DST)
    use_cl = cfg.CONTINUAL.continual and cfg.CONTINUAL.method.name == 'DST'
    
    # Check if the model architecture type is convolution-based or transformer-based
    # More robust check - look for specific architecture attributes
    is_conv_model = False
    if hasattr(model, 'take_layer') and callable(getattr(model, 'take_layer')):
        if hasattr(model, 'layers_names') and len(getattr(model, 'layers_names', [])) > 0:
            # Has all the required attributes and methods for convolution models
            is_conv_model = True
            _logger.info("Detected convolution-based architecture with SpaceNet support")
        else:
            _logger.warning("Model has take_layer but missing layers_names attribute. Treating as transformer architecture.")
    else:
        _logger.info("Detected transformer-based architecture")
        
    # For transformer models, ensure the problematic attributes are properly stubbed
    # This is to avoid errors when CL code tries to access conv-specific attributes
    if not is_conv_model:
        # Add dummy take_layer that always returns False to avoid masking non-existent attributes
        if not hasattr(model, 'take_layer'):
            model.take_layer = lambda name, param: False
        
        # Add empty layers_names to avoid indexing errors
        if not hasattr(model, 'layers_names'):
            model.layers_names = []

    if use_cl:
        if is_conv_model:
            # Initialize for convolution model
            _logger.info("Initializing CL_Transformer for convolution architecture")
            
            # Get required parameters for convolution model from config
            freezed_nodes_count_perlayer = cfg.CONTINUAL.method.get('freezed_nodes_count_perlayer', [])
            num_selected_nodes = cfg.CONTINUAL.method.get('num_selected_nodes', [])
            num_additional_selected_nodes = cfg.CONTINUAL.method.get('num_additional_selected_nodes', [])
            no_neurons_reused_from_previous = cfg.CONTINUAL.method.get('no_neurons_reused_from_previous', [])
            
            # Ensure these parameters are actually defined
            if not freezed_nodes_count_perlayer or not num_selected_nodes or not num_additional_selected_nodes or not no_neurons_reused_from_previous:
                _logger.warning("Missing required convolution parameters, falling back to transformer approach")
                # Fall back to transformer approach
                cl_manager = CL_Transformer(
                    model=model, 
                    device=accelerator.device, 
                    sparsity_config=sparsity_config, 
                    replace_percentage=0.2
                )
            else:
                # Initialize with convolution-specific parameters
                cl_manager = CL_Transformer(
                    model=model, 
                    device=accelerator.device, 
                    sparsity_config=sparsity_config,
                    task_labels=task_labels,
                    freezed_nodes_count_perlayer=freezed_nodes_count_perlayer,
                    num_selected_nodes=num_selected_nodes,
                    num_additional_selected_nodes=num_additional_selected_nodes,
                    no_neurons_reused_from_previous=no_neurons_reused_from_previous,
                    replace_percentage=0.2
                )
        else:
            # Initialize for transformer model (simpler approach)
            _logger.info("Initializing CL_Transformer for transformer architecture")
            cl_manager = CL_Transformer(
                model=model, 
                device=accelerator.device, 
                sparsity_config=sparsity_config, 
                replace_percentage=0.2
            )
            
        # Verify the CL_Transformer object has all necessary methods
        required_methods = ['drop', 'grow', 'prepare_next_task', 'save_current_mask', 'set_evaluation_mask']
        has_all_methods = True
        for method in required_methods:
            if not hasattr(cl_manager, method):
                has_all_methods = False
                _logger.error(f"CL_Transformer missing required method: {method}")
        
        if not has_all_methods:
            _logger.error("CL_Transformer is missing required methods. This will cause errors during training.")
            if input("Continue anyway? (y/n): ").lower() != 'y':
                import sys
                sys.exit(1)
            
        cl_manager.set_init_network_weight()
    else:
        cl_manager = None
    
    epoch_time_m = AverageMeter()
    end = time.time() 
    
    for n_task, (current_class_name, class_loader_dict) in enumerate(loader_dict.items()):

        if n_task == 0:
            model.reconstruction.transformer.classification_head.set_initial_task(len(current_class_name))
        else:
            model.reconstruction.transformer.classification_head.expand(len(current_class_name))
        
        # Reset importance at the start of each task
        if use_cl:
            cl_manager.reset_importance()
        
        best_score = 0.0
        if (n_task == 0) or (cfg.CONTINUAL.continual==False):
            drift_monitor = DriftMonitor(log_dir=os.path.join(savedir,'DriftMonitor.log'))
        
        torch.cuda.empty_cache()
        _logger.info(f"Current Class Name : {current_class_name}")        
            
        # Init optimizer & scheduler 
        optimizer = __import__('torch.optim',fromlist='optim').__dict__[cfg.OPTIMIZER.opt_name](model.parameters(), lr=cfg.OPTIMIZER.lr, **cfg.OPTIMIZER.params)        
        if cfg.SCHEDULER.name is not None:                        
            scheduler = __import__('torch.optim.lr_scheduler', fromlist='lr_scheduler').__dict__[cfg.SCHEDULER.name](optimizer, **cfg.SCHEDULER.params)
        else:
            scheduler = None
            
        # Init dataloader 
        trainloader, testloader = class_loader_dict['train'],class_loader_dict['test']
        
        model, trainloader, testloader, optimizer, scheduler = accelerator.prepare(model, trainloader, testloader, optimizer, scheduler)
        
        model = model.cuda()
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
                    cfg          = cfg,
                    drift_monitor= drift_monitor,
                    cl_manager   = cl_manager
                )
             
            if scheduler:
                scheduler.step()
                
                epoch_time_m.update(time.time() - end)
                end = time.time()
                
            if use_cl and epoch < epochs - 1:
                try:
                    # Check if methods exist before calling
                    if hasattr(cl_manager, 'drop') and hasattr(cl_manager, 'grow'):
                        # Try to drop and grow connections
                        cl_manager.drop()
                        cl_manager.grow()
                    else:
                        if not hasattr(cl_manager, 'drop'):
                            _logger.error("CL_Transformer object missing 'drop' method")
                        if not hasattr(cl_manager, 'grow'):
                            _logger.error("CL_Transformer object missing 'grow' method")
                        _logger.warning("Skipping drop/grow due to missing methods")
                except Exception as e:
                    _logger.error(f"Error during drop/grow operations: {e}")
                    _logger.warning("Skipping drop/grow for this epoch.")
                
            if (epoch%5 == 0) or (epoch%199 == 0): 
                # Get metric_list from configuration if available
                metric_list = cfg.DATASET.get('metric_list', None) if hasattr(cfg.DATASET, 'metric_list') else None
                
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
                    metric_list        = metric_list
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
            
            # prepare evaluation 
            if use_cl:
                try:
                    if hasattr(cl_manager, 'save_current_mask'):
                        cl_manager.save_current_mask()
                    else:
                        _logger.error("CL_Transformer object missing 'save_current_mask' method")

                    if hasattr(cl_manager, 'set_evaluation_mask'):
                        cl_manager.set_evaluation_mask()
                    else:
                        _logger.error("CL_Transformer object missing 'set_evaluation_mask' method")
                except Exception as e:
                    _logger.error(f"Error in evaluation preparation: {e}")
            
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
                
                # Get metric_list from configuration if available
                metric_list = cfg.DATASET.get('metric_list', None) if hasattr(cfg.DATASET, 'metric_list') else None
                
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
                                last               = True,
                                metric_list        = metric_list
                            )
            if n_task < len(loader_dict) - 1 and use_cl:
                # Create a safe wrapper for prepare_next_task with parameters
                def safe_prepare_next_task(cl_manager, is_conv_model):
                    if not hasattr(cl_manager, 'prepare_next_task'):
                        _logger.error("CL_Transformer object missing 'prepare_next_task' method")
                        # Basic fallback - just increment task counter
                        if hasattr(cl_manager, 'current_task'):
                            cl_manager.current_task += 1
                        return False
                    
                    try:
                        # Determine if we need class relation-based masks for convolution models
                        if is_conv_model and hasattr(cfg.CONTINUAL.method, 'selection_method'):
                            # For convolution-based models with class relation approach
                            selection_method = cfg.CONTINUAL.method.get('selection_method', 'random')
                            enable_reuse = cfg.CONTINUAL.method.get('enable_reuse', False)
                            
                            # Get task representation if available
                            t2_representation = None
                            if hasattr(cfg.CONTINUAL, 'task_representation'):
                                t2_representation = cfg.CONTINUAL.task_representation
                            
                            _logger.info(f"Preparing next task with selection method: {selection_method}, enable_reuse: {enable_reuse}")
                            return cl_manager.prepare_next_task(
                                selection_method_for_related_class=selection_method, 
                                enable_reuse=enable_reuse, 
                                t2_representation=t2_representation
                            )
                        else:
                            # For transformer models or simple convolution models
                            _logger.info("Preparing next task with standard approach")
                            return cl_manager.prepare_next_task()
                    except Exception as e:
                        _logger.error(f"Error during prepare_next_task: {e}")
                        _logger.warning("Attempting basic task transition...")
                        # Try a very basic approach as fallback
                        if hasattr(cl_manager, 'current_task'):
                            cl_manager.current_task += 1
                        return False
                
                # Call the safe wrapper
                safe_prepare_next_task(cl_manager, is_conv_model)

                
        else:
            model  = __import__('models').__dict__[cfg.MODEL.method](
                backbone    = cfg.MODEL.backbone,
                **cfg.MODEL.params
                ).cuda()
            model = accelerator.prepare(model)
            _logger.info('Model init')