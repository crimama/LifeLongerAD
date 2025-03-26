import os 
from PIL import Image 
from arguments import parser 
import torch 
import torch.nn as nn 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
from datasets import create_dataset 
from torch.utils.data import DataLoader
from utils.metrics import MetricCalculator, loco_auroc
from accelerate import Accelerator
from omegaconf import OmegaConf
import seaborn as sns 

def get_features(trainloader,model):
    featureloader = model.get_feature_loader(trainloader)
    features = [] 
    for feat, target in featureloader:
        features.append(feat.detach().cpu().numpy())            
    features = np.vstack(features)
    
    proxy, _ = model.core.featuresampler.run(features)
    model.set_criterion(proxy)
    
    return features , featureloader 

def get_projected(cls_name, model, featureloader):
    model.eval()
    model = model.to('cuda')
    weight = torch.load(f'/Volume/LifeLongerAD_cu121/results/ProxyCore/MVTecAD/Nsoftmax-focal_loss-Continual_False-online_False/seed_42/model_weight/{cls_name}_model.pth')
    model.load_state_dict(weight)
    
    projected = [] 
    for feat, target in featureloader:
        with torch.no_grad():
            proj = model.embedding_layer(feat.to('cuda'))
            
        projected.append(proj.detach().cpu().numpy())       
        
    projected = np.vstack(projected)
    return projected 

def tsne_reduction(X):
    from openTSNE import TSNE
    from sklearn.decomposition import PCA
    import numpy as np

    # ▶️ fast t-SNE (openTSNE)
    tsne = TSNE(n_jobs=8, random_state=42, perplexity=30, n_iter=300)
    X_tsne = tsne.fit(X)
    return X_tsne 

if __name__=='__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '7' 
    default_setting = './configs/default/mvtecad.yaml'
    model_setting = './configs/model/proxycore.yaml'
    cfg = parser(True,default_setting, model_setting)


    model  = __import__('models').__dict__[cfg.MODEL.method](
            backbone = cfg.MODEL.backbone,
            **cfg.MODEL.params
            ).to('cuda')
    device = 'cuda'


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
        
    tsne_dir = '/Volume/LifeLongerAD_cu121/tsne'

    from tqdm import tqdm 
    for cls_name in tqdm(cfg.DATASET.class_names):
        print(cls_name)
        trainloader = loader_dict[cls_name]['train']
        features, featureloader = get_features(trainloader,model)
        
        projected = get_projected(cls_name, model, featureloader)
        
        
        reduced_features = tsne_reduction(features)
        np.save(os.path.join(tsne_dir,f'{cls_name}_features.npy'),
                reduced_features)


        reduced_projected = tsne_reduction(projected)
        np.save(os.path.join(tsne_dir,f'{cls_name}_projected.npy'),
                reduced_projected)
        
        break 