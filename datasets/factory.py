import os
import pandas as pd
from glob import glob
from torchvision import datasets
from .augmentation import train_augmentation, test_augmentation, gt_augmentation
from .mvtecad import MVTecAD
from .visa import VISA

def create_dataset(dataset_name: str, datadir: str, class_name: str, img_size: int, mean: list, std: list, aug_info: bool = None, **params):
    trainset, testset = eval(f"load_{dataset_name}")(
                                                    dataset_name = dataset_name,
                                                    datadir      = datadir,
                                                    class_name   = class_name,
                                                    img_size     = img_size,
                                                    mean         = mean,
                                                    std          = std,
                                                    aug_info     = aug_info,
                                                    **params
                                                )
    return trainset, testset

def load_VISA(dataset_name:str, datadir:str, class_name:str, img_size:int, mean:list, std:list, aug_info = None, baseline: bool = False, anomaly_ratio: float = 0.0):
    df = get_visa_df(
            dataset_name  = dataset_name,
            datadir       = datadir,
            class_name    = class_name,
            baseline      = baseline,
            anomaly_ratio = anomaly_ratio
        )

    trainset = VISA(
                df           = df,
                train_mode   = 'train',
                transform    = train_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
                gt_transform = gt_augmentation(img_size = img_size, aug_info = aug_info),
                gt           = True 
            )

    testset = VISA(
                df           = df,
                train_mode   = 'test',
                transform    = test_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
                gt_transform = gt_augmentation(img_size = img_size, aug_info = aug_info),
                gt           = True 
            )
    
    return trainset, testset 

def load_MVTecAD(dataset_name:str, datadir:str, class_name:str, img_size:int, mean:list, std:list, aug_info = None, baseline: bool = False, anomaly_ratio: float = 0.0):
    df = get_df(
            dataset_name  = dataset_name,
            datadir       = datadir,
            class_name    = class_name,
            baseline      = baseline,
            anomaly_ratio = anomaly_ratio
        )

    trainset = MVTecAD(
                df           = df,
                class_name   = class_name, 
                train_mode   = 'train',
                transform    = train_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
                gt_transform = gt_augmentation(img_size = img_size, aug_info = aug_info)
            )

    testset = MVTecAD(
                df           = df,
                class_name   = class_name, 
                train_mode   = 'test',
                transform    = test_augmentation(img_size = img_size, mean = mean, std = std, aug_info = aug_info),
                gt_transform = gt_augmentation(img_size = img_size, aug_info = aug_info)
            )
    
    return trainset, testset 

def get_visa_df(dataset_name: str, datadir: str, class_name: str, baseline: bool = True, anomaly_ratio: float = 0.0):
    visa_dirs = pd.read_csv(os.path.join(datadir, dataset_name, 'split_csv/1cls.csv'))
    visa_dirs = visa_dirs[visa_dirs['object'] == class_name].reset_index(drop=True)
    visa_dirs = visa_dirs.rename(columns={'split': 'train/test', 'label': 'anomaly', 'image': 0})
    visa_dirs['anomaly'] = visa_dirs['anomaly'].apply(lambda x: 0 if x == 'normal' else 1)
    visa_dirs[0] = visa_dirs[0].apply(lambda x: os.path.join(datadir, 'VISA', x))
    visa_dirs['mask'] = visa_dirs['mask'].apply(lambda x: os.path.join(datadir, 'VISA', str(x)))

    if not baseline:
        visa_dirs = train_test_split(df=visa_dirs, max_ratio=0.05, anomaly_ratio=anomaly_ratio)
    
    return visa_dirs

def get_df(dataset_name: str, datadir: str, class_name: str, baseline: bool = True, anomaly_ratio: float = 0.0):
    img_dirs = get_img_dirs(dataset_name, datadir, class_name)
    img_dirs['train/test'] = img_dirs[0].apply(lambda x: x.split('/')[-3])
    
    if not baseline:
        img_dirs = train_test_split(df=img_dirs, anomaly_ratio=anomaly_ratio)
    
    return img_dirs

def get_img_dirs(dataset_name: str, datadir: str, class_name: str) -> pd.DataFrame:
    class_name = '*' if class_name == 'all' else class_name
    img_paths = sorted(glob(os.path.join(datadir, dataset_name, class_name, '*/*/*.png')))
    img_dirs = pd.DataFrame([p for p in img_paths if 'ground_truth' not in p], columns=[0]).reset_index(drop=True)
    img_dirs['anomaly'] = img_dirs[0].apply(lambda x: 1 if x.split('/')[-2] != 'good' else 0)
    img_dirs['train/test'] = ''
    return img_dirs

def train_test_split(df, max_ratio=0.2, anomaly_ratio=0.2):
    num_train = len(df[df['train/test'] == 'train'])
    num_max_anomaly = int(num_train * max_ratio)
    num_anomaly_train = int(num_train * anomaly_ratio)

    unfix_anomaly_index = df[(df['train/test'] == 'test') & (df['anomaly'] == 1)].sample(num_max_anomaly).index
    df.loc[unfix_anomaly_index, 'train/test'] = 'unfix'

    fix_test = df[df['train/test'] == 'test'].reset_index(drop=True)

    notuse_train_normal = df[df['train/test'] == 'train'].sample(num_anomaly_train).index
    df.loc[notuse_train_normal, 'train/test'] = 'notuse'

    train_anomaly_index = df[df['train/test'] == 'unfix'].sample(num_anomaly_train).index
    df.loc[train_anomaly_index, 'train/test'] = 'train'

    fix_train = df[df['train/test'] == 'train'].reset_index(drop=True)

    final_df = pd.concat([fix_train, fix_test]).reset_index(drop=True)
    
    return final_df