import numpy as np
import os
import random
import wandb
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
    os.makedirs(savedir, exist_ok=True)    
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
                # class_names = cfg.DATASET.class_names, #Todo 여기 부분 다른 모델도 맞춰서 수정 필요 
                **cfg.MODEL.params
                )
        
    #! LOAD DATASET FOR CONTINUAL LEARNING 
    loader_dict = {}
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
    # class_names = ['wood','cable','chewinggum','grid','pill','pcb2','macaroni2','pcb4','candle','tile','pcb1','pcb3','capsule','fryum','transistor','cashew','metal_nut','carpet','bottle','zipper','pipe_fryum','toothbrush','capsules','leather','hazelnut','screw','macaroni1']
    # dataset = {'wood': 'MVTecAD', 'cable': 'MVTecAD', 'chewinggum': 'VISA', 'grid': 'MVTecAD', 'pill': 'MVTecAD', 'pcb2': 'VISA', 'macaroni2': 'VISA', 'pcb4': 'VISA', 'candle': 'VISA', 'tile': 'MVTecAD', 'pcb1': 'VISA', 'pcb3': 'VISA', 'capsule': 'MVTecAD', 'fryum': 'VISA', 'transistor': 'MVTecAD', 'cashew': 'VISA', 'metal_nut': 'MVTecAD', 'carpet': 'MVTecAD', 'bottle': 'MVTecAD', 'zipper': 'MVTecAD', 'pipe_fryum': 'VISA', 'toothbrush': 'MVTecAD', 'capsules': 'VISA', 'leather': 'MVTecAD', 'hazelnut': 'MVTecAD', 'screw': 'MVTecAD', 'macaroni1': 'VISA'}
    # for cn in class_names:
    #     trainset, testset = create_dataset(
    #         dataset_name  = dataset[cn],
    #         datadir       = cfg.DATASET.datadir,
    #         class_name    = cn,
    #         img_size      = cfg.DATASET.img_size,
    #         mean          = cfg.DATASET.mean,
    #         std           = cfg.DATASET.std,
    #         aug_info      = cfg.DATASET.aug_info,
    #         **cfg.DATASET.get('params',{})
    #     )
        trainloader = DataLoader(
            dataset     = trainset,
            batch_size  = cfg.DATASET.batch_size,
            num_workers = cfg.DATASET.num_workers,
            shuffle     = True 
        )    

        testloader = DataLoader(
                dataset     = testset,
                batch_size  = cfg.DATASET.batch_size,
                num_workers = cfg.DATASET.num_workers,
                shuffle     = False 
            )    
        
        loader_dict[cn] = {'train':trainloader,'test':testloader}
    
    
    # optimizer 
    # if cfg.OPTIMIZER.name is not None:    
    
    if cfg.TRAIN.wandb.use:
        wandb.init(name=f'{cfg.DEFAULT.exp_name}', project=cfg.TRAIN.wandb.project_name, config=OmegaConf.to_container(cfg))

                
        
    __import__(f'train.train_{cfg.MODEL.method.lower()}', fromlist=f'train_{cfg.MODEL.method.lower()}').fit(
            model        = model, 
            loader_dict  = loader_dict,
            accelerator  = accelerator,
            epochs       = cfg.TRAIN.epochs, 
            use_wandb    = cfg.TRAIN.wandb.use,
            log_interval = cfg.TRAIN.log_interval,
            savedir      = savedir,
            seed         = cfg.DEFAULT.seed,
            cfg          = cfg
        )     
    

if __name__=='__main__':
    
    # config
    cfg = parser()
    
    # run
    run(cfg)