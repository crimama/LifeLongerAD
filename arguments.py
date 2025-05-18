from omegaconf import OmegaConf
import argparse
from datasets import stats
from easydict import EasyDict
import os 
import importlib 

def convert_type(value):
    # None
    if value == 'None':
        return None
    
    # list or tuple
    elif len(value.split(',')) > 1:
        return value.split(',')
    
    # bool
    check, value = str_to_bool(value)
    if check:
        return value
    
    # float
    check, value = str_to_float(value)
    if check:
        return value
    
    # int
    check, value = str_to_int(value)
    if check:
        return value
    
    return value

def str_to_bool(value):
    try:
        check = isinstance(eval(value), bool)
        out = [True, eval(value)] if check else [False, value]
        return out
    except NameError:
        return False, value
    
def str_to_float(value):
    try:
        check = isinstance(eval(value), float)
        out = [True, eval(value)] if check else [False, value]
        return out
    except NameError:
        return False, value
    
def str_to_int(value):
    try:
        check = isinstance(eval(value), int)
        out = [True, eval(value)] if check else [False, value]
        return out
    except NameError:
        return False, value
    
def cli_parser():
    args = OmegaConf.from_cli()
    default_cfg = OmegaConf.load(args.default_setting)
    model_cfg = OmegaConf.load(args.model_setting)
    cfg = OmegaConf.merge(default_cfg,model_cfg)
    cfg = OmegaConf.merge(cfg,args)

    return cfg 

def jupyter_parser(default_setting:str = None, model_setting:str = None):
    default = OmegaConf.load(default_setting)
    model = OmegaConf.load(model_setting)
    return OmegaConf.merge(default, model)

#! PARSER 
def parser(jupyter:bool = False, default_setting:str = None, model_setting:str = None):
    
    if jupyter:
        cfg = jupyter_parser(default_setting, model_setting)
    else:
        cfg = cli_parser()
    
               
    # load dataset statistics
    if cfg.DATASET.dataset_name in ['MVTecAD','MVTecLoco','VISA','BTAD','MPDD']:
        cfg.DATASET.update(stats.datasets['ImageNet'])
    else:    
        cfg.DATASET.update(stats.datasets[cfg.DATASET.dataset_name])    
    
    # update config for each method 
    
    if cfg.MODEL.method in ['PatchCore','ReconPatch','ProxyCoreBase','SoftPatch','CoreInit']:
        cfg = patchcore_arguments(cfg)
    else:
        pass
    
    #! Continual Setting         
    if cfg.CONTINUAL.online:
        cfg.TRAIN.epochs = 1
        cfg.DATASET.batch_size = 1                
    
    # Update experiment name
    # if cfg.MODEL.method in ['PatchCore','SoftPatch']:
    #     # cfg.DEFAULT.exp_name = f"{cfg.DEFAULT.exp_name}-coreset_ratio_{cfg.MODEL.params.coreset_sampling_ratio}-anomaly_ratio_{cfg.DATASET.params.anomaly_ratio}" 
    #     cfg.DEFAULT.exp_name = f"{cfg.DEFAULT.exp_name}-{cfg.MODEL.params.weight_method}-sampling_ratio_{cfg.MODEL.params.sampling_ratio}" 
    # else:
    #     cfg.DEFAULT.exp_name = f"{cfg.DEFAULT.exp_name}-online_{cfg.CONTINUAL.online}-unified_{cfg.CONTINUAL.unified}-init_ratio_{cfg.CONTINUAL.init_data_ratio}-nb_tasks_{cfg.CONTINUAL.nb_tasks}"
    
    cfg.DEFAULT.exp_name = f"{cfg.DEFAULT.exp_name}-Continual_{cfg.CONTINUAL.continual}-online_{cfg.CONTINUAL.online}"
    
    # IUF Config update 
    if cfg.MODEL.method in ['IUF','UniADBuilder','CFGCAD','CAD']:
        cfg = iuf_config_update(cfg)
        
    
    # Print Experiment name 
    print(f"\n Experiment Name : {cfg.DEFAULT.exp_name}\n")            
    return cfg  

def patchcore_arguments(cfg):
    # device for Patchcore 
    if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
        cfg.MODEL.params.device = f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', None)}"
    else:
        cfg.MODEL.params.device = 'cuda:0'    
        
    return cfg 


def uniad_update_config(config):
    # update planes & strides
    net_cfg = config.MODEL.params.net_cfg
    backbone_path, backbone_type = net_cfg[0].type.rsplit(".", 1)
    module = importlib.import_module(backbone_path)
    backbone_info = getattr(module, "backbone_info")
    backbone = backbone_info[backbone_type]
    outplanes = []
    for layer in net_cfg[0].kwargs.outlayers:
        if layer not in backbone["layers"]:
            raise ValueError(
                "only layer {} for backbone {} is allowed, but get {}!".format(
                    backbone["layers"], backbone_type, layer
                )
            )
        idx = backbone["layers"].index(layer)
        outplanes.append(backbone["planes"][idx])

    net_cfg[2].kwargs.instrides = net_cfg[1].kwargs.outstrides
    net_cfg[2].kwargs.inplanes = [sum(outplanes)]
    
    config.MODEL.params.net_cfg = net_cfg
    return config


def iuf_config_update(config):
    # update feature size
    _, reconstruction_type = config.MODEL.params.net_cfg[2].type.rsplit(".", 1)
    if reconstruction_type in ["UniAD","CFGReconstruction"]:
        input_size = [config.DATASET.img_size for i in range(2)]
        outstride = config.MODEL.params.net_cfg[1].kwargs.outstrides[0]
        assert (
            input_size[0] % outstride == 0
        ), "input_size must could be divided by outstrides exactly!"
        assert (
            input_size[1] % outstride == 0
        ), "input_size must could be divided by outstrides exactly!"
        feature_size = [s // outstride for s in input_size]        
        config.MODEL.params.net_cfg[2].kwargs.feature_size = feature_size

    # update planes & strides
    backbone_path, backbone_type = config.MODEL.params.net_cfg[0].type.rsplit(".", 1)
    module = importlib.import_module(backbone_path)
    backbone_info = getattr(module, "backbone_info")
    backbone = backbone_info[backbone_type]
    outblocks = None
    if "efficientnet" in backbone_type:
        outblocks = []
    outstrides = []
    outplanes = []
    for layer in config.MODEL.params.net_cfg[0].kwargs.outlayers:
        if layer not in backbone["layers"]:
            raise ValueError(
                "only layer {} for backbone {} is allowed, but get {}!".format(
                    backbone["layers"], backbone_type, layer
                )
            )
        idx = backbone["layers"].index(layer)
        if "efficientnet" in backbone_type:
            outblocks.append(backbone["blocks"][idx])
        outstrides.append(backbone["strides"][idx])
        outplanes.append(backbone["planes"][idx])
    if "efficientnet" in backbone_type:
        config.MODEL.params.net_cfg[0].kwargs.pop("outlayers")
        config.MODEL.params.net_cfg[0].kwargs.outblocks = outblocks
    config.MODEL.params.net_cfg[0].kwargs.outstrides = outstrides
    config.MODEL.params.net_cfg[1].kwargs.outplanes = sum(outplanes)
    
    # if "MVTecAD" == config.DATASET.dataset_name:
    #     config.MODEL.params.net_cfg[2].kwargs.num_classes

    return config
