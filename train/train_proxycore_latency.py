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
from utils.log import AverageMeter, metric_logging, DriftMonitor
import warnings
warnings.filterwarnings('ignore')

_logger = logging.getLogger('train')

# ----------------------------
# thop 라이브러리에서 profile 함수 import
# ----------------------------
try:
    from thop import profile
except ImportError:
    profile = None
    _logger.warning("thop 패키지가 설치되어 있지 않아 FLOPs 계산을 건너뜁니다. 'pip install thop'으로 설치할 수 있습니다.")

def train(model, dataloader, featureloader, testloader, optimizer, scheduler, accelerator, log_interval: int, epoch, epochs, savedir, cfg, drift_monitor) -> dict:
    
    def collect_gradients(cfg, model, all_gradients, epoch, step):
        if ((cfg.CONTINUAL.online and (step % 10 == 0)) or (not cfg.CONTINUAL.online and ((epoch) % 2 == 0) and (step % 4 == 0))):
            step_grad_dict = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    step_grad_dict[name] = param.grad.clone().detach().cpu().numpy()
            all_gradients.append(step_grad_dict)

    # ----------------------------------------------------------
    # model 인자를 추가하여 FLOPs 측정까지 진행할 수 있도록 수정
    # ----------------------------------------------------------
    def log_training_info(step, accelerator, dataloader, epoch, epochs, losses_m, optimizer, batch_time_m, data_time_m, feats, model):
        # GPU 메모리(MB) 측정
        gpu_mem_usage = torch.cuda.memory_allocated(device=accelerator.device) / (1024**2)  # MB 단위

        # Latency (ms)
        latency_ms = batch_time_m.avg * 1000.0

        # Throughput (img/s)
        throughput = feats.size(0) / batch_time_m.avg if batch_time_m.avg > 0 else 0.0

        # FLOPs 측정 (thop.profile)
        # log_interval마다 한 번씩만 측정
        if profile is not None:
            flops_val, params = profile(
                model,
                inputs=(feats.to(accelerator.device),),
                verbose=False
            )
            flops_val = flops_val / 1e9  # GFLOPs 단위로 변환
        else:
            flops_val = 0.0

        _logger.info(
            'Train Epoch [{epoch}/{epochs}] [{cur_step}/{total_step}] '
            'Total Loss: {loss.avg:>6.4f} '
            'LR: {lr:.3e} '
            'Time: {batch_time.avg:.3f}s, {rate_avg:>3.2f}/s '
            'Data: {data_time.avg:.3f}s '
            '| FLOPs: {flops_val:.2f} GFLOPs '
            '| GPU Mem: {gpu_mem:.2f} MB '
            '| Latency: {latency:.2f} ms '
            '| Throughput: {throughput:.2f} img/s'.format(
                (step + 1) // accelerator.gradient_accumulation_steps,
                len(dataloader) // accelerator.gradient_accumulation_steps,
                epoch=epoch,
                epochs=epochs,
                loss=losses_m,
                lr=optimizer.param_groups[0]['lr'],
                batch_time=batch_time_m,
                data_time=data_time_m,
                rate_avg=feats.size(0) / batch_time_m.avg if batch_time_m.avg > 0 else 0.0,
                flops_val=flops_val,
                gpu_mem=gpu_mem_usage,
                latency=latency_ms,
                throughput=throughput,
                cur_step=(step + 1),
                total_step=len(dataloader)
            )
        )

    def do_online_inference(cfg, featureloader, model, accelerator, savedir, epoch, testloader, current_class_name, batch_time_m, optimizer):
        if (epoch==3):
            test_metrics = test(
                model=model, featureloader=featureloader, testloader=testloader, device=accelerator.device, 
                savedir=savedir, use_wandb=False,
                epoch=epoch, optimizer=optimizer,
                epoch_time_m=batch_time_m, class_name=current_class_name,
                current_class_name=current_class_name
            )

    def save_gradients_if_needed(cfg, dataloader, epoch, savedir, all_gradients):
        if (cfg.CONTINUAL.online or (not cfg.CONTINUAL.online and ((epoch) % 2 == 0))):
            current_class_name_ = dataloader.dataset.class_name
            np.save(f"{savedir}/gradients/{current_class_name_}_gradient_log_epoch_{epoch}.npy", all_gradients)

    batch_time_m = AverageMeter(); data_time_m = AverageMeter(); losses_m = AverageMeter()
    current_class_name = dataloader.dataset.class_name
    end = time.time()
    
    model.train()
    all_gradients = []
    for step, (feat, target) in enumerate(featureloader):
        data_time_m.update(time.time() - end)
        
        if cfg.DATASET.embed_augemtation:
            feat = feat + torch.randn(feat.shape).to(feat.device)        
        
        # predict
        outputs = model(feat.to(accelerator.device)) # outputs = [z,w]
        loss   = model.criterion([outputs, target])
        outputs.retain_grad() # for reverse distillation         

        optimizer.zero_grad()
        accelerator.backward(loss)

        losses_m.update(loss.item())
        #* collect_gradients(cfg, model, all_gradients, epoch, step)
        drift_monitor.update(outputs) # RD의 경우 decoder feature를 추출하기 위해 outputs[1][1] 사용         
        optimizer.step()
        
        batch_time_m.update(time.time() - end)
        
        adjusted_log_interval = log_interval if cfg.CONTINUAL.online else 10

        # -------------------------------------------------------------------------
        # log_interval마다 FLOPs, GPU 메모리, Latency, Throughput을 추가로 logging
        # -------------------------------------------------------------------------
        if (step + 1) % adjusted_log_interval == 0:
            log_training_info(
                step=step,
                accelerator=accelerator,
                dataloader=featureloader,  # featureloader의 전체 길이를 파악하기 위해
                epoch=epoch,
                epochs=epochs,
                losses_m=losses_m,
                optimizer=optimizer,
                batch_time_m=batch_time_m,
                data_time_m=data_time_m,
                feats=feat,   # 현재 배치를 FLOPs 측정 등에 활용
                model=model   # 모델 자체를 넘겨서 thop.profile에 사용
            )

        end = time.time()
    
    #! save_gradients_if_needed(cfg, dataloader, epoch, savedir, all_gradients)    
    return {"loss": losses_m.avg, "gradients": all_gradients, "class_name": current_class_name}



def test(model, featureloader, testloader, device,
         savedir, use_wandb, epoch, optimizer, epoch_time_m, class_name, current_class_name,
         last : bool = False) -> dict:
    
    for name, module in model.named_modules():
        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear() 
            
    from utils.metrics import MetricCalculator, loco_auroc    
    model.eval()
    img_level = MetricCalculator(metric_list = ['auroc','average_precision'])
    pix_level = MetricCalculator(metric_list = ['auroc','average_precision'])     

    target_oriented_train_feat = [] 
    for feat, target in featureloader:
        feat = feat.to(device)
        with torch.no_grad():
            z = model.embedding_layer(feat)
            target_oriented_train_feat.append(z.detach().cpu().numpy())            
    target_oriented_train_feat = np.vstack(target_oriented_train_feat)
    
    sample_features, _ = model.core.featuresampler.run(target_oriented_train_feat)
    model.core.anomaly_scorer.fit(detection_features=[sample_features])

    #! Inference     
    for step, (images, labels, cls, gts) in enumerate(testloader):

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
            ) # Unfold : (B)
                        
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            masks = model.core.anomaly_segmentor.convert_to_segmentation(patch_scores) # interpolation : (B,pw,ph) -> (B,W,H)
                        
            score_map = np.concatenate([np.expand_dims(sm,0) for sm in masks])
            score_map = np.expand_dims(score_map,1)
            
            # get image wise anomaly score 
            image_scores = model.core.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = model.core.patch_maker.score(image_scores)
                
        # Stack Scoring for metrics 
        pix_level.update(score_map,gts.type(torch.int))
        img_level.update(image_scores, labels.type(torch.int))
            
    i_results, p_results = img_level.compute(), pix_level.compute()
    _logger.info(f"Current Class name : {current_class_name} Class name : {class_name} "
                 f"Image AUROC: {i_results['auroc']:.3f} | Pixel AUROC: {p_results['auroc']:.3f}")
        
    test_result = OrderedDict(img_level = i_results)
    test_result.update([('pix_level', p_results)])            
    
    
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

        return featureloader
    
    for n_task, (current_class_name, class_loader_dict) in enumerate(loader_dict.items()):
        if (n_task == 0) or (cfg.CONTINUAL.continual==False):
            drift_monitor = DriftMonitor(log_dir=os.path.join(savedir,'DriftMonitor.log'))
        
        class_name, class_loader_dict = list(loader_dict.items())[n_task]
        trainloader, testloader = loader_dict[class_name]['train'],loader_dict[class_name]['test']
        
        featureloader = generate_proxy_and_labels(model, trainloader, accelerator,cfg)
        
        torch.cuda.empty_cache()
        _logger.info(f"Current Class Name : {current_class_name}")        
        
        # Init optimizer & Scheduler 
        from adamp import AdamP 
        optimizer = AdamP(model.parameters(), lr=cfg.OPTIMIZER.lr, **cfg.OPTIMIZER.params)
        
        if cfg.SCHEDULER.name is not None:            
            scheduler = __import__('torch.optim.lr_scheduler', fromlist='lr_scheduler').__dict__[cfg.SCHEDULER.name](optimizer, **cfg.SCHEDULER.params)
        else:
            scheduler = None
            
        # Init Dataloader 
        trainloader, testloader = loader_dict[current_class_name]['train'],loader_dict[current_class_name]['test']
        
        model, featureloader, trainloader, testloader, optimizer, scheduler = accelerator.prepare(model, featureloader, trainloader, testloader, optimizer, scheduler)
        
        # Train 
        for epoch in range(epochs):
            train(
                model           = model, 
                dataloader      = trainloader, 
                featureloader   = featureloader, 
                testloader      = testloader, 
                optimizer       = optimizer, 
                scheduler       = scheduler,
                accelerator     = accelerator, 
                log_interval    = log_interval,
                epoch           = epoch,
                epochs          = epochs,
                savedir         = savedir, 
                cfg             = cfg,
                drift_monitor   = drift_monitor 
            )            
            if scheduler:
                scheduler.step()
                epoch_time_m.update(time.time() - end)
                end = time.time()
                
            if (epoch % 200 == 0) or (epoch % 1990 == 0): 
                test_metrics = test(
                    model              = model, 
                    featureloader      = featureloader, 
                    testloader         = testloader, 
                    device             = accelerator.device, 
                    savedir            = savedir, 
                    use_wandb          = False,
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
            # Continual evaluation 
            num_start = 0 
            num_end = num_current_class+2 if num_current_class != len(loader_dict)-1 else len(loader_dict)

            for n_task in range(num_start,num_end):
                class_name, class_loader_dict = list(loader_dict.items())[n_task]
                trainloader, testloader = loader_dict[class_name]['train'],loader_dict[class_name]['test']
                trainloader, testloader = accelerator.prepare(trainloader, testloader)            
                
                test_metrics = test(
                    model              = model, 
                    featureloader      = featureloader,
                    device             = accelerator.device,
                    savedir            = savedir, 
                    use_wandb          = use_wandb,
                    epoch              = 0 if epochs == 0 else epoch,
                    optimizer          = optimizer, 
                    epoch_time_m       = epoch_time_m,
                    class_name         = trainloader.dataset.class_name,
                    current_class_name = current_class_name,
                    testloader         = testloader,
                    last               = True
                )
        else:
            model  = __import__('models').__dict__[cfg.MODEL.method](
                backbone    = cfg.MODEL.backbone,
                **cfg.MODEL.params
            )
            model = accelerator.prepare(model)
            _logger.info('Model init')
