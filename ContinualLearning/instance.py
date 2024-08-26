from torch.utils.data import Subset 
from copy import deepcopy
import random 
import pandas as pd 
import numpy as np 

class InstanceIncremental(object):
    def __init__(self,trainset, scheduler, optimizer,
                 batch_size, epochs,
                 nb_tasks, init_data_ratio):
        self.trainset = trainset 
        self.nb_tasks = nb_tasks 
        self.init_data_ratio = init_data_ratio         
        
        self.scheduler = scheduler 
        self.optimizer = optimizer 
        
        self.trainsets = self.__data_split__(trainset, nb_tasks, init_data_ratio)
        self.n_epoch_task = self.set_epochs(epochs,batch_size)
        
        self.auc_result = {} # key : t model, value : test result on t model 
        self.fm_result = {} # key : t model, value : task(train) inference result on t model  
        self.task_model_list = [] 
        
    def set_epochs(self, epochs, batch_size):
        import math 
        n_data_task = np.array([len(t) for t in self.trainsets])
        n_data_task_p = n_data_task/np.sum(n_data_task)

        total_step = (math.ceil(len(self.trainset)/batch_size)) * epochs 
        n_step_task = n_data_task_p * total_step 
        n_epoch_task = n_step_task  / np.ceil(n_data_task/batch_size)        
        return n_epoch_task.astype(np.uint8)
        
        
        
    def __data_split__(self, trainset, nb_tasks, init_data_ratio):
        # trainset index 생성 
        indices = list(range(len(trainset)))

        # 초기 학습 데이터 비율 샘플링 
        sample_size = int(len(indices) * init_data_ratio)
        first_index_list = random.sample(indices, sample_size)

        # 남은 인덱스 구하기
        remaining_indices = list(set(indices) - set(first_index_list))

        # 남은 인덱스를 9개의 리스트로 랜덤하게 분배
        random.shuffle(remaining_indices)
        split_indices = [remaining_indices[i::int(nb_tasks-1)] for i in range(nb_tasks-1)]

        index_list = [first_index_list]
        index_list.extend(split_indices)

        trainsets = [
            Subset(trainset, idxs) for idxs in index_list
            ]
        if self.nb_tasks == 1:
            trainsets = [Subset(trainset,first_index_list)]
            return trainsets
        else:
            return trainsets 
    
    def __call__(self,task_index, model=None):        
        
        if model is not None:
            optimizer = __import__('torch.optim',fromlist='optim').__dict__[self.optimizer.opt_name](model.parameters(), lr=self.optimizer.lr, **self.optimizer.params)
            
            if self.scheduler.name is not None:
                scheduler = __import__('torch.optim.lr_scheduler', fromlist='lr_scheduler').__dict__[self.scheduler.name](optimizer, **self.scheduler.params)
                
            else:
                scheduler = None 
                
            trainset = self.trainsets[task_index]
            epoch_task = self.n_epoch_task[task_index]
            
            return trainset, optimizer, scheduler, epoch_task
        else: 
            trainset = self.trainsets[task_index]
            return trainset 
        
    def eval(self):
        fm = pd.DataFrame(self.fm_result)
        fm_result = [] 

            
        if self.nb_tasks != 1:
            auc_result = pd.DataFrame()
            for j in range(0,self.nb_tasks-1):
                
                # for forgetting measure : (a_ji - a_jk) / a_ji
                fm_result.append(
                    (min(fm.loc[j][:-1]- fm.loc[j][self.nb_tasks-1])) / fm.loc[j][self.nb_tasks-1]
                    )
                
                # for auc result 
                j_result = pd.DataFrame(self.auc_result[j]).reset_index().melt(id_vars=['index'])
                j_result['task'] = j 
                auc_result = pd.concat([auc_result,j_result])
            
            fm = np.mean(fm_result)    
        else:
            fm = 0 
            
        # calculate bwt 
        # 이전 task round 대비 성능 변화율 
        # 성능 변화 그래프의 평균 기울기라 볼 수 있음 
        
        if self.nb_tasks != 1:
            auc_result = auc_result.reset_index(drop=True)
            bwt_result={}
            for level in ['img_level','pix_level']:
                bwt_result[level] = {} 
                for metric in ['auroc','average_precision']:
                    auc_result_ml = auc_result[(auc_result['index']==metric) & (auc_result['variable']==level)]
                    bwt = np.mean((auc_result_ml['value'].iloc[1:].values - auc_result_ml['value'].iloc[:-1].values)/auc_result_ml['value'].iloc[:-1].values)
                    bwt_result[level][metric] = bwt 
        else:
            bwt_result = 0 
                
        return {'fm':fm, 'bwt':bwt_result}
            
        