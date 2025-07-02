import logging
import time
import os 
import numpy as np

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from collections import OrderedDict

from utils.metrics import MetricCalculator
from utils.log import AverageMeter, metric_logging, save_performance_summary, generate_performance_report
import warnings
warnings.filterwarnings('ignore')

# Import for performance monitoring
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("thop not available. FLOPs calculation will be skipped.")

# Efficient metrics logging flag
ENABLE_PERFORMANCE_MONITORING = False  # Set to False for efficient metrics logging

_logger = logging.getLogger('train')

# Performance monitoring utility class
class PerformanceMonitor:
    def __init__(self, device, enabled=True):
        self.device = device
        self.enabled = enabled
        self.reset()
    
    def reset(self):
        self.total_samples = 0
        self.total_time = 0.0
        self.max_memory = 0.0
        self.flops_calculated = False
        self.flops = 0
        self.params = 0
        
    def start_batch(self):
        if not self.enabled:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.batch_start_time = time.time()
        
    def end_batch(self, batch_size):
        if not self.enabled:
            return
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
        if not self.enabled or not THOP_AVAILABLE or self.flops_calculated:
            return
        try:
            # Try to get the unwrapped model if it's wrapped by accelerator
            if hasattr(model, 'module'):
                model_for_flops = model.module
            else:
                model_for_flops = model
            
            # Create a temporary copy and move to CPU to avoid device conflicts
            input_copy = input_tensor.cpu() if input_tensor.is_cuda else input_tensor
            
            # Set model to eval mode temporarily for FLOPs calculation
            original_training = model_for_flops.training
            model_for_flops.eval()
            
            # Create a simple wrapper to avoid hook issues
            import copy
            model_copy = copy.deepcopy(model_for_flops).cpu()
            model_copy.eval()
            
            flops, params = profile(model_copy, inputs=(input_copy,), verbose=False)
            self.flops = flops
            self.params = params
            self.flops_calculated = True
            
            # Clean up
            del model_copy
            
            # Restore original training mode
            if original_training:
                model_for_flops.train()
            else:
                model_for_flops.eval()
                
        except Exception as e:
            if self.enabled:
                _logger.warning(f"FLOPs calculation failed: {e}")
            # Try alternative method with simpler input
            try:
                # Use a minimal tensor for calculation
                if hasattr(model, 'module'):
                    model_for_flops = model.module
                else:
                    model_for_flops = model
                
                # Just count parameters as fallback
                self.params = sum(p.numel() for p in model_for_flops.parameters())
                self.flops = 0  # Set to 0 if calculation fails
                self.flops_calculated = True
                if self.enabled:
                    _logger.info(f"FLOPs calculation failed, but counted {self.params} parameters")
            except Exception as e2:
                if self.enabled:
                    _logger.warning(f"Parameter counting also failed: {e2}")
                self.flops = 0
                self.params = 0
                self.flops_calculated = True  # Mark as calculated to avoid repeated attempts
    
    def get_metrics(self):
        if not self.enabled:
            return {}
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
        if not self.enabled:
            return {}
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
    

def train(model, dataloader, featureloader, optimizer, accelerator, log_interval: int, epoch, epochs, cfg, savedir) -> dict:
    
    def log_training_info(step, accelerator, dataloader, epoch, epochs, losses_m, optimizer, batch_time_m, data_time_m, feats, perf_monitor):
        if ENABLE_PERFORMANCE_MONITORING:
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
                    rate_avg=feats.size(0) / batch_time_m.avg, data_time=data_time_m,
                    throughput=metrics['throughput_samples_per_sec'],
                    gpu_mem=metrics['max_gpu_memory_gb']
                )
            )
        else:
            _logger.info(
                'Train Epoch [{epoch}/{epochs}] " [{:d}/{}] Total Loss: {loss.avg:>6.4f} '
                'LR: {lr:.3e} '
                'Time: {batch_time.avg:.3f}s, {rate_avg:>3.2f}/s '
                'Data: {data_time.avg:.3f}s'.format(
                    (step + 1) // accelerator.gradient_accumulation_steps,
                    len(dataloader) // accelerator.gradient_accumulation_steps,
                    epoch=epoch, epochs=epochs, loss=losses_m,
                    lr=optimizer.param_groups[0]['lr'], batch_time=batch_time_m,
                    rate_avg=feats.size(0) / batch_time_m.avg, data_time=data_time_m
                )
            )
    
    # Initialize performance monitor
    perf_monitor = PerformanceMonitor(accelerator.device, enabled=ENABLE_PERFORMANCE_MONITORING)
    
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    current_class_name = dataloader.dataset.class_name
    end = time.time()
    
    # Calculate FLOPs using a sample from featureloader
    if ENABLE_PERFORMANCE_MONITORING:
        try:
            sample_batch = next(iter(featureloader))
            sample_feat = sample_batch[0]
            if len(sample_feat) > 0:
                # Use a safe approach for FLOPs calculation
                sample_input = sample_feat[:1]
                perf_monitor.calculate_flops(model, sample_input)
        except Exception as e:
            _logger.warning(f"Failed to calculate FLOPs during initialization: {e}")
            # Continue without FLOPs calculation
    
    model.train()
    for step, (feat, target) in enumerate(featureloader):
        # Start performance monitoring
        perf_monitor.start_batch()
        
        data_time_m.update(time.time() - end)
        
        if cfg.DATASET.embed_augemtation:
            feat = feat + torch.randn(feat.shape).to(feat.device)        
        
        # predict
        outputs = model(feat.to(accelerator.device))
        loss = model.criterion([outputs, target])
        outputs.retain_grad() # for reverse distillation         
        
        optimizer.zero_grad()
        accelerator.backward(loss)
        
        losses_m.update(loss.item())
        optimizer.step()
        
        batch_time_m.update(time.time() - end)
        
        # End performance monitoring
        perf_monitor.end_batch(feat.size(0))
        
        adjusted_log_interval = log_interval if cfg.CONTINUAL.online else 10
        if (step + 1) % adjusted_log_interval == 0:
            log_training_info(step, accelerator, featureloader, epoch, epochs, losses_m, optimizer, batch_time_m, data_time_m, feat, perf_monitor)
        
        end = time.time()
    
    # Log training performance metrics
    train_metrics = {}
    if ENABLE_PERFORMANCE_MONITORING:
        train_metrics = perf_monitor.log_metrics("Training")
        
        # Save performance metrics
        os.makedirs(f"{savedir}/performance_logs", exist_ok=True)
        with open(f"{savedir}/performance_logs/{current_class_name}_train_epoch_{epoch}_performance.txt", "w") as f:
            for key, value in train_metrics.items():
                f.write(f"{key}: {value}\n")

        # Log training summary with performance metrics
        _logger.info(f"Training completed for {current_class_name} - Epoch {epoch}")
        _logger.info(f"Total samples processed: {train_metrics['total_samples']}")
        _logger.info(f"Training throughput: {train_metrics['throughput_samples_per_sec']:.2f} samples/sec")
        _logger.info(f"Max GPU Memory usage: {train_metrics['max_gpu_memory_gb']:.3f} GB")
    else:
        _logger.info(f"Training completed for {current_class_name} - Epoch {epoch}")
    
    return {
        "loss": losses_m.avg, 
        "class_name": current_class_name,
        "performance_metrics": train_metrics
    }


def test(model, featureloader, testloader, device,
         savedir, use_wandb, epoch, optimizer, epoch_time_m, class_name, current_class_name,
         last: bool = False) -> dict:
    
    # Initialize performance monitor for inference
    perf_monitor = PerformanceMonitor(device, enabled=ENABLE_PERFORMANCE_MONITORING)
    
    model.eval()
    img_level = MetricCalculator(metric_list=['auroc', 'average_precision'])
    pix_level = MetricCalculator(metric_list=['auroc', 'average_precision'])     

    target_oriented_train_feat = [] 
    for feat, target in featureloader:
        feat = feat.to(device)
        with torch.no_grad():
            z = model.embedding_layer(feat)
            target_oriented_train_feat.append(z.detach().cpu().numpy())            
    target_oriented_train_feat = np.vstack(target_oriented_train_feat)
    
    sample_features, _ = model.core.featuresampler.run(target_oriented_train_feat)
    model.core.anomaly_scorer.fit(detection_features=[sample_features])

    # Inference     
    for step, (images, labels, cls, gts) in enumerate(testloader):
        # Start performance monitoring
        perf_monitor.start_batch()
        
        _ = model.core.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            # create features of test images 
            features, patch_shapes = model.core._embed(images.to(device), provide_patch_shapes=True)
            features = torch.Tensor(np.vstack(features)).to(device)        
            features = model.embedding_layer(features)
            
            # predict anomaly score 
            image_scores, _, _ = model.core.anomaly_scorer.predict([features.detach().cpu().numpy()])            
            
            # get patch wise anomaly score using image score    
            patch_scores = model.core.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize 
            )
                        
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            masks = model.core.anomaly_segmentor.convert_to_segmentation(patch_scores)
                        
            score_map = np.concatenate([np.expand_dims(sm, 0) for sm in masks])
            score_map = np.expand_dims(score_map, 1)
            
            # get image wise anomaly score 
            image_scores = model.core.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = model.core.patch_maker.score(image_scores)
        
        # Calculate FLOPs on first batch
        if step == 0 and ENABLE_PERFORMANCE_MONITORING:
            try:
                # Use a more robust approach for FLOPs calculation
                sample_input = images[:1].cpu()  # Move to CPU and use single sample
                
                # Try to get the unwrapped model if it's wrapped by accelerator
                if hasattr(model, 'module'):
                    model_for_flops = model.module
                else:
                    model_for_flops = model
                
                # Set model to eval mode temporarily for FLOPs calculation
                original_training = model_for_flops.training
                model_for_flops.eval()
                
                perf_monitor.calculate_flops(model_for_flops, sample_input)
                
                # Restore original training mode
                if original_training:
                    model_for_flops.train()
                else:
                    model_for_flops.eval()
                    
            except Exception as e:
                _logger.warning(f"FLOPs calculation failed during inference: {e}")
                # Continue without FLOPs calculation
        
        # End performance monitoring
        perf_monitor.end_batch(images.size(0))
                
        # Stack Scoring for metrics 
        pix_level.update(score_map, gts.type(torch.int))
        img_level.update(image_scores, labels.type(torch.int))
    
    # Log inference performance metrics
    inference_metrics = {}
    if ENABLE_PERFORMANCE_MONITORING:
        inference_metrics = perf_monitor.log_metrics("Inference")
        
        # Save inference performance metrics
        os.makedirs(f"{savedir}/performance_logs", exist_ok=True)
        with open(f"{savedir}/performance_logs/{class_name}_test_epoch_{epoch}_performance.txt", "w") as f:
            for key, value in inference_metrics.items():
                f.write(f"{key}: {value}\n")
            
    i_results, p_results = img_level.compute(), pix_level.compute()
    _logger.info(f"Current Class name : {current_class_name} Class name : {class_name} Image AUROC: {i_results['auroc']:.3f}| Pixel AUROC: {p_results['auroc']:.3f}")
        
    test_result = OrderedDict(img_level=i_results)
    test_result.update([('pix_level', p_results)])
    if ENABLE_PERFORMANCE_MONITORING:
        test_result.update([('inference_performance', inference_metrics)])
    
    metric_logging(
        savedir=savedir, use_wandb=use_wandb, epoch=epoch,
        optimizer=optimizer, epoch_time_m=epoch_time_m,
        test_metrics=test_result,
        class_name=class_name, current_class_name=current_class_name,
        **{'last': last}
    )
    return test_result 


def fit(
    model, loader_dict: dict, accelerator,
    epochs: int, use_wandb: bool, log_interval: int, eval_interval: int, seed: int = None, savedir: str = None,
    cfg=None):
    
    print(savedir)
    best_score = 0.0
    epoch_time_m = AverageMeter()
    end = time.time() 
    
    def generate_proxy_and_labels(model, trainloader, accelerator, cfg):
        featureloader = accelerator.prepare(model.get_feature_loader(trainloader))
        features = []
        for feat, _ in featureloader:
            features.append(feat.detach().cpu().numpy())
        features = np.vstack(features)

        model.core.featuresampler.percentage = cfg.MODEL.params.pslabel_sampling_ratio
        proxy, _ = model.core.featuresampler.run(features)
        model.core.featuresampler.percentage = cfg.MODEL.params.sampling_ratio

        proxy = nn.functional.normalize(torch.Tensor(proxy), dim=1)
        proxy_label = []
        for feat, _ in featureloader:
            proxy_label.append(torch.matmul(feat, proxy.T.to(feat.device)).argmax(dim=1))
        proxy_label = torch.concat(proxy_label)
        
        featureloader.dataset.labels = proxy_label
        model.set_criterion(proxy)

        # Clean up intermediate variables
        del features, proxy, proxy_label
        torch.cuda.empty_cache()

        return featureloader
    
    for n_task, (current_class_name, class_loader_dict) in enumerate(loader_dict.items()):
        best_score = 0.0        
            
        trainloader, testloader = class_loader_dict['train'], class_loader_dict['test']
        featureloader = generate_proxy_and_labels(model, trainloader, accelerator, cfg)
        
        torch.cuda.empty_cache()
        _logger.info(f"Current Class Name : {current_class_name}")        
            
        # Init optimizer & Scheduler 
        from adamp import AdamP 
        optimizer = AdamP(model.parameters(), lr=cfg.OPTIMIZER.lr, **cfg.OPTIMIZER.params)
        
        if cfg.SCHEDULER.name is not None:            
            scheduler = __import__('torch.optim.lr_scheduler', fromlist='lr_scheduler').__dict__[cfg.SCHEDULER.name](optimizer, **cfg.SCHEDULER.params)
        else:
            scheduler = None
            
        model, featureloader, trainloader, testloader, optimizer, scheduler = accelerator.prepare(model, featureloader, trainloader, testloader, optimizer, scheduler)
        
        # Calculate model FLOPs and parameters using sample input
        model_flops, model_params = 0.0, 0.0
        if ENABLE_PERFORMANCE_MONITORING:
            try:
                # For feature-based model, use feature tensor size
                sample_feat = next(iter(featureloader))[0]
                input_size = sample_feat[:1].shape  # Use single sample shape
                
                # Try to get the unwrapped model if it's wrapped by accelerator
                if hasattr(model, 'module'):
                    model_for_flops = model.module
                else:
                    model_for_flops = model
                
                # Set model to eval mode temporarily
                original_training = model_for_flops.training
                model_for_flops.eval()
                
                # Create a deep copy to avoid hook issues
                import copy
                model_copy = copy.deepcopy(model_for_flops).cpu()
                model_copy.eval()
                
                # Move input to CPU to avoid device conflicts
                input_copy = sample_feat[:1].cpu()
                
                model_flops, model_params = profile(model_copy, inputs=(input_copy,), verbose=False)
                _logger.info(f"Model FLOPs: {model_flops / 1e9:.2f} GFLOPs, Parameters: {model_params / 1e6:.2f} M")
                
                # Clean up
                del model_copy
                
                # Restore original training mode
                if original_training:
                    model_for_flops.train()
                else:
                    model_for_flops.eval()
                    
            except Exception as e:
                _logger.warning(f"FLOPs calculation failed: {e}")
                # Fallback to parameter counting only
                try:
                    if hasattr(model, 'module'):
                        model_for_flops = model.module
                    else:
                        model_for_flops = model
                    model_params = sum(p.numel() for p in model_for_flops.parameters())
                    model_flops = 0.0
                    _logger.info(f"FLOPs calculation failed, but counted {model_params / 1e6:.2f} M parameters")
                except Exception as e2:
                    _logger.warning(f"Parameter counting also failed: {e2}")
                    model_flops = 0.0
                    model_params = 0.0
        
        # Train 
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # FPS measurement start
            start_time = time.time()
            total_processed_samples = 0
            
            train_result = train(
                model=model, 
                dataloader=trainloader, 
                featureloader=featureloader, 
                optimizer=optimizer, 
                accelerator=accelerator, 
                log_interval=log_interval,
                epoch=epoch,
                epochs=epochs,
                cfg=cfg,
                savedir=savedir
            )
            
            # FPS measurement end
            if ENABLE_PERFORMANCE_MONITORING:
                end_time = time.time()
                epoch_duration = end_time - epoch_start_time
                num_batches = len(featureloader)
                batch_size = next(iter(featureloader))[0].size(0) * accelerator.num_processes
                total_processed_samples = num_batches * batch_size
                fps = total_processed_samples / epoch_duration

                # Calculate FLOPs per batch (GFLOPs)        
                
                flops_per_batch = model_flops / num_batches if num_batches > 0 else 0            

                _logger.info(f"Epoch [{epoch}/{epochs}] - FPS: {fps:.2f}, Estimated FLOPs per batch: {flops_per_batch} GFLOPs")
            
            if scheduler:
                scheduler.step()
                
            epoch_time_m.update(time.time() - end)
            end = time.time()
                
            if (epoch % 20 == 0) or (epoch % 199 == 0): 
                test_metrics = test(
                    model=model, 
                    featureloader=featureloader, 
                    testloader=testloader, 
                    device=accelerator.device, 
                    savedir=savedir, 
                    use_wandb=False,
                    epoch=epoch, 
                    optimizer=optimizer,
                    epoch_time_m=epoch_time_m, 
                    class_name=current_class_name,
                    current_class_name=current_class_name
                )

        # EVALUATION
        num_current_class = list(loader_dict.keys()).index(current_class_name)
        
        score = (test_metrics['img_level']['average_precision'] + test_metrics['pix_level']['average_precision']) / 2
        if best_score < score:
            os.makedirs(f"{savedir}/model_weight/", exist_ok=True)
            torch.save(model.state_dict(), f"{savedir}/model_weight/{current_class_name}_model.pth")
            best_score = score 

        if cfg.CONTINUAL.continual:
            # Continual evaluation 
            num_start = 0 
            num_end = num_current_class + 2 if num_current_class != len(loader_dict) - 1 else len(loader_dict)

            for n_task in range(num_start, num_end):
                class_name, class_loader_dict = list(loader_dict.items())[n_task]
                trainloader, testloader = loader_dict[class_name]['train'], loader_dict[class_name]['test']
                trainloader, testloader = accelerator.prepare(trainloader, testloader)            
                
                test_metrics = test(
                    model=model, 
                    featureloader=featureloader,
                    device=accelerator.device,
                    savedir=savedir, 
                    use_wandb=use_wandb,
                    epoch=0 if epochs == 0 else epoch,
                    optimizer=optimizer, 
                    epoch_time_m=epoch_time_m,
                    class_name=trainloader.dataset.class_name,
                    current_class_name=current_class_name,
                    testloader=testloader,
                    last=True
                )
                
                # Save performance summary for continual learning
                if ENABLE_PERFORMANCE_MONITORING and 'performance_metrics' in train_result and 'inference_performance' in test_metrics:
                    save_performance_summary(
                        savedir=savedir,
                        class_name=class_name,
                        train_metrics=train_result['performance_metrics'],
                        test_metrics=test_metrics['inference_performance'],
                        epoch=epoch
                    )
        else:
            # For non-continual learning, also save performance summary
            if ENABLE_PERFORMANCE_MONITORING and 'performance_metrics' in train_result and 'inference_performance' in test_metrics:
                save_performance_summary(
                    savedir=savedir,
                    class_name=current_class_name,
                    train_metrics=train_result['performance_metrics'],
                    test_metrics=test_metrics['inference_performance'],
                    epoch=epoch-1 if epoch > 0 else 0
                )
            
            # Clear model memory before recreating
            model.clear_memory()
            
            # Delete the old model completely before creating new one
            del model
            torch.cuda.empty_cache()
            
            model = __import__('models').__dict__[cfg.MODEL.method](
                backbone=cfg.MODEL.backbone,
                **cfg.MODEL.params
            )
            model = accelerator.prepare(model)
            _logger.info('Model init')
        
        # Clean up memory at the end of each task
        del featureloader, trainloader, testloader, optimizer
        if scheduler is not None:
            del scheduler
        torch.cuda.empty_cache()
        _logger.info(f"Task {current_class_name} completed. GPU memory cleared.")
    
    # Generate final performance report only if performance monitoring is enabled
    if ENABLE_PERFORMANCE_MONITORING:
        _logger.info("Generating comprehensive performance report...")
        generate_performance_report(savedir)
    else:
        _logger.info("Performance monitoring disabled - skipping comprehensive performance report generation")