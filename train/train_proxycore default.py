import logging
import time
import os 
import numpy as np

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from collections import OrderedDict

from utils.metrics import MetricCalculator
from utils.log import AverageMeter, metric_logging
import warnings
warnings.filterwarnings('ignore')

_logger = logging.getLogger('train')
    

def train(model, dataloader, featureloader, optimizer, accelerator, log_interval: int, epoch, epochs, cfg) -> dict:
    
    def log_training_info(step, accelerator, dataloader, epoch, epochs, losses_m, optimizer, batch_time_m, data_time_m, feats):
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
    
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    current_class_name = dataloader.dataset.class_name
    end = time.time()
    
    model.train()
    for step, (feat, target) in enumerate(featureloader):
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
        
        adjusted_log_interval = log_interval if cfg.CONTINUAL.online else 10
        if (step + 1) % adjusted_log_interval == 0:
            log_training_info(step, accelerator, featureloader, epoch, epochs, losses_m, optimizer, batch_time_m, data_time_m, feat)
        
        end = time.time()
    
    return {"loss": losses_m.avg, "class_name": current_class_name}


def test(model, featureloader, testloader, device,
         savedir, use_wandb, epoch, optimizer, epoch_time_m, class_name, current_class_name,
         last: bool = False) -> dict:
    
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
                
        # Stack Scoring for metrics 
        pix_level.update(score_map, gts.type(torch.int))
        img_level.update(image_scores, labels.type(torch.int))
            
    i_results, p_results = img_level.compute(), pix_level.compute()
    _logger.info(f"Current Class name : {current_class_name} Class name : {class_name} Image AUROC: {i_results['auroc']:.3f}| Pixel AUROC: {p_results['auroc']:.3f}")
        
    test_result = OrderedDict(img_level=i_results)
    test_result.update([('pix_level', p_results)])            
    
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
        
        # Train 
        for epoch in range(epochs):
            train(
                model=model, 
                dataloader=trainloader, 
                featureloader=featureloader, 
                optimizer=optimizer, 
                accelerator=accelerator, 
                log_interval=log_interval,
                epoch=epoch,
                epochs=epochs,
                cfg=cfg
            )            
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
        else:
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