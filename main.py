import numpy as np
import pandas as pd 
import os
import random
import wandb
import json 
import torch
import logging
from arguments import parser

from datasets import create_dataset
from utils.log import setup_default_logging

from accelerate import Accelerator
from omegaconf import OmegaConf

from torch.utils.data import DataLoader
from adamp import AdamP
import warnings
warnings.filterwarnings('ignore')

torch.autograd.set_detect_anomaly(True)

_logger = logging.getLogger('train')

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)


def run(cfg):
    #! DIRECTORY SETTING 
    savedir = os.path.join(
                            cfg.DEFAULT.savedir,
                            cfg.MODEL.method,
                            cfg.DATASET.dataset_name,
                            cfg.DEFAULT.exp_name,
                            f"seed_{cfg.DEFAULT.seed}"
                            )    
    
    # make directory 
    os.makedirs(savedir, exist_ok=True)    
    os.makedirs(os.path.join(savedir,'results'), exist_ok=True)
    os.makedirs(os.path.join(savedir, 'gradients'), exist_ok=True)
    os.makedirs(os.path.join(savedir, 'model_weight'), exist_ok=True)
    OmegaConf.save(cfg, os.path.join(savedir, 'configs.yaml'))
    setup_default_logging(log_path=os.path.join(savedir,'train.log'))
    
    #! TRAIN SETTING 
    # set accelerator
    accelerator = Accelerator(
        mixed_precision             = cfg.TRAIN.mixed_precision
    )    
    
    # set seed 
    torch_seed(cfg.DEFAULT.seed)

    # set device
    _logger.info('Device: {}'.format(accelerator.device))

    # model     
    model  = __import__('models').__dict__[cfg.MODEL.method](
                backbone    = cfg.MODEL.backbone,
                **cfg.MODEL.params
                )
    
    #! LOAD DATASET FOR CONTINUAL LEARNING     
    loader_dict = {}
    accelerator = Accelerator()
    for cn in cfg.DATASET.class_names:
        trainset, testset = create_dataset(
            dataset_name  = cfg.DATASET.dataset_name,
            datadir       = cfg.DATASET.datadir,
            class_name    = cn,
            img_size      = cfg.DATASET.img_size,
            mean          = cfg.DATASET.mean,
            std           = cfg.DATASET.std,
            aug_info      = cfg.DATASET.aug_info,
            **cfg.DATASET.get('params',{})
        )
        trainloader = DataLoader(
            dataset     = trainset,
            batch_size  = cfg.DATASET.batch_size,
            num_workers = cfg.DATASET.num_workers,
            shuffle     = True 
        )    

        testloader = DataLoader(
                dataset     = testset,
                batch_size  = 8,
                num_workers = cfg.DATASET.num_workers,
                shuffle     = False 
            ) 
        
        loader_dict[cn] = {'train':trainloader,'test':testloader}
    
    
    # optimizer 
    # if cfg.OPTIMIZER.name is not None:    
    if cfg.TRAIN.wandb.use:
        wandb.init(name=f'{cfg.DEFAULT.exp_name}', project=cfg.TRAIN.wandb.project_name, config=OmegaConf.to_container(cfg))


                
    
    if cfg.MODEL.method == 'IUF':
        TRAINER = __import__(f'train.train_{cfg.MODEL.method.lower()}', fromlist=f'train_{cfg.MODEL.method.lower()}').fit
        
    else: 
        from train.train import fit 
        TRAINER = fit
    
    TRAINER(
            model         = model, 
            loader_dict   = loader_dict,
            accelerator   = accelerator,
            epochs        = cfg.TRAIN.epochs, 
            use_wandb     = cfg.TRAIN.wandb.use,
            log_interval  = cfg.TRAIN.log_interval,
            eval_interval = cfg.TRAIN.eval_interval,
            savedir       = savedir,
            seed          = cfg.DEFAULT.seed,
            cfg           = cfg
        )
    
    # final evaluation & save
    
    
def metric_save(cfg, continual:bool = True):
    from utils.metrics import compute_continual_result
    savedir = os.path.join(
                            cfg.DEFAULT.savedir,
                            cfg.MODEL.method,
                            cfg.DATASET.dataset_name,
                            cfg.DEFAULT.exp_name,
                            f"seed_{cfg.DEFAULT.seed}",
                            )
    # load results
    results = pd.read_csv(os.path.join(savedir,'results/result_log.csv'))
    main_result, forgetting_result = compute_continual_result(results, continual=continual)
    
    # save main result, forgetting result 
    main_result.to_csv(os.path.join(savedir,'results','main.csv'))
    forgetting_result.to_csv(os.path.join(savedir,'results','forgetting.csv'))
    
        
if __name__=='__main__':
    
    # config
    cfg = parser()

    # run
    run(cfg)
    
    metric_save(cfg,continual=cfg.CONTINUAL.continual)