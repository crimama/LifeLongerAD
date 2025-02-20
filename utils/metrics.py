import pandas as pd 
import numpy as np 
from skimage import measure
from scipy.ndimage.measurements import label
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve, roc_curve, auc, confusion_matrix
from torchmetrics import AUROC 
from statistics import mean
import torch 

def compute_continual_result(result: pd.DataFrame, 
                              metric_list: list = ['img_level_auroc','img_level_auroc','img_level_average_precision','pix_level_average_precision'],
                              continual: bool = True) -> tuple:
    """
    주어진 결과 DataFrame에서 최종 결과(main_result)와 잊어버림(forgetting) 지표를 계산합니다.
    
    Parameters:
        result (pd.DataFrame): 결과 데이터프레임. 
            반드시 'last', 'GT_class_name', 'class_name', 'task', 'epoch', 'epoch_time' 열이 포함되어 있어야 합니다.
        metric_list (list, optional): 계산할 메트릭 이름 리스트. 
            기본값은 ['img_level_auroc','img_level_auroc','img_level_average_precision','pix_level_average_precision'] 입니다.
    
    Returns:
        main_result (pd.DataFrame): 최종 결과 DataFrame (각 메트릭 평균 포함)
        forgetting_result (pd.DataFrame): 클래스별 Forgetting, BWT, FWT 결과 DataFrame
    """        
    # === Last result 및 AA 계산 ===
    last_class = result['GT_class_name'].iloc[-1]
    class_name_list = result['class_name'].unique()
    
    if continual:
        main_result = result.loc[(result['last'] == 1) & (result['GT_class_name'] == last_class)] \
                            .reset_index(drop=True) \
                            .drop(columns=['task', 'epoch', 'GT_class_name', 'epoch_time'])
        
        # AA: class_name, last 열을 제외한 열의 평균값 계산 후 마지막 행에 추가
        aa_values = main_result.drop(columns=['class_name', 'last']).mean()
        aa_values['class_name'] = 'AA'
        main_result = pd.concat([main_result, pd.DataFrame(aa_values).T], ignore_index=True)
    
        # === Forgetting, BWT, FWT 계산 ===
        
        forgetting_result = pd.DataFrame()        
        # 첫 클래스는 제외 (예: 첫 클래스는 기준이므로)
        for cln in class_name_list[1:]:
            # 현재 클래스에 해당하는 데이터 추출 (불필요한 열 제거)
            temp = result[result['class_name'] == cln].drop(columns=['task', 'epoch', 'epoch_time'])
            
            # 마지막 결과 행 추출 (last == 1)
            last_value = temp[temp['last'] == 1].iloc[-1]
            
            # Average Forgetting (AF): 
            # last==0 인 경우의 수치형 값 최대치와 last_value의 차이를 계산
            max_value = temp[temp['last'] == 0].select_dtypes(include=['number']).max()
            AF = max_value - last_value
            
            # Backward Transfer (BWT):
            # 현재 클래스에 해당하는 GT_class_name을 가진 행에서 계산.
            bwt_value = temp[(temp['last'] == 1) & (temp['GT_class_name'] == cln)] \
                            .drop(columns=['class_name', 'GT_class_name'])
            # last_value와 bwt_value의 차이에서 마지막 행 값 사용            
            BWT = (last_value - bwt_value)
            
            # Forward Transfer (FWT):
            # last==1인 경우 첫 행에서 계산.
            fwt_value = temp[temp['last'] == 1] \
                            .drop(columns=['class_name', 'GT_class_name']).iloc[0]
            FWT = (bwt_value - fwt_value)
            
            # 각 메트릭에 대해 AF, BWT, FWT 값을 DataFrame에 정리
            # eval 대신 직접 인덱싱하여 값을 추출합니다.
            data = {
                'AF': [AF[m] for m in metric_list],
                'BWT': [BWT[m] for m in metric_list],
                'FWT': [FWT[m] for m in metric_list]
            }
            df_metrics = pd.DataFrame(data, index=metric_list).reset_index().rename(columns={'index': 'metric'})
            df_metrics['class_name'] = cln
            
            forgetting_result = pd.concat([forgetting_result, df_metrics], ignore_index=True)
                
    else:        
        main_result = pd.DataFrame([result[result['GT_class_name'] == cln].iloc[-1] for cln in class_name_list]).drop(columns=['task','epoch','class_name','last','epoch_time']).reset_index(drop=True)
        forgetting_result =  pd.DataFrame()
    return main_result, forgetting_result
    

def loco_auroc(metric, dataloader):
    results = {} 
    metric.preds = np.concatenate(metric.preds)
    metric.targets = np.concatenate(metric.targets)
    
    anomaly_types = pd.Series(dataloader.dataset.img_dirs).apply(lambda x : x.split('/')[-2])
    
    for a_type in ['structural_anomalies','logical_anomalies']:
        cond = np.where(anomaly_types.apply(lambda x : x in ['good',a_type]))[0]
        
        preds =  metric.preds[cond]
        target = metric.targets[cond]
        
        auroc = roc_auc_score(target.reshape(-1),preds.reshape(-1))
        results[a_type] = auroc 
    return results 

class MetricCalculator:
    '''
    metric = MetricCalculator(metric_list=['auroc
    
    metric.update(preds:torch.Tensor, target:torch.Tensor)
    '''
    def __init__(self, metric_list: list):
        self.metric_list = metric_list 
        self.preds = [] 
        self.targets = [] 
        
    def _confusion_matrix(self, y_preds:np.ndarray, y_trues:np.ndarray):
        y_preds, y_trues = y_preds.flatten(), y_trues.flatten()
        fpr, tpr, thr = roc_curve(y_trues,y_preds)
        cm_list = {} 
        for fpr_ratio in [0.005, 0.01, 0.05, 0.1]:
            fpr_thr = thr[np.argmin(fpr < fpr_ratio)-1]
            y_preds = np.where(y_preds > fpr_thr,1,0)
            cm = confusion_matrix(y_trues, y_preds)
            
            cm_list[fpr_ratio] = cm.tolist() 
        return cm_list 
    
    def _detach(self, data):
        if isinstance(data, torch.Tensor):
            out = data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            out = data 
        return out 
    
    def _average_precision(self, y_preds:np.ndarray, y_trues:np.ndarray):
        y_preds, y_trues = y_preds.flatten(), y_trues.flatten()
        precision, recall, _ = precision_recall_curve(y_trues, y_preds)
        aupr = auc(recall, precision)
        return aupr 
    
    def _auroc(self, y_preds:np.ndarray, y_trues:np.ndarray):
        y_preds, y_trues = y_preds.flatten(), y_trues.flatten()
        fpr, tpr, thr = roc_curve(y_trues,y_preds)
        auroc = auc(fpr,tpr)
        return auroc
    
    def _aupro(self, y_preds:np.ndarray, y_trues:np.ndarray):
        fprs, pros = calculate_aupro(y_preds, y_trues)
        aupro = auc(fprs,pros)
        return aupro 
    
    def update(self, y_preds:torch.Tensor, y_trues:torch.Tensor) -> None:
        '''
        preds : torch.Tensor -> np.ndarray -> append in list 
        '''
        self.preds.append(
            self._detach(y_preds)
        )
        self.targets.append(
            self._detach(y_trues)
        )        
        
    def compute(self) -> dict:
        y_preds = np.concatenate(self.preds) # (N,W,H)
        y_trues = np.concatenate(self.targets) # (N,W,H)
        
        result_list = {}         
        for metric in self.metric_list:
            result_list[metric] = eval(f'self._{metric}')(y_preds, y_trues)            
        return result_list 

def calculate_aupro(anomaly_maps:np.ndarray, ground_truth_maps:np.ndarray):    
    # 모든 이미지에 대한 FPR과 PRO 값을 저장할 리스트
    all_fprs = []
    all_pros = []

    # 각 이미지에 대해 compute_pro 호출
    for i in range(len(anomaly_maps)):
        fprs, pros = compute_pro(anomaly_maps[i], ground_truth_maps[i])
        all_fprs.extend(fprs[1:-1])  # 첫 번째와 마지막 요소(0과 1)는 제외
        all_pros.extend(pros[1:-1])  # 첫 번째와 마지막 요소(0과 1)는 제외

    # 전체 FPR과 PRO 값들을 정렬
    sorted_indices = np.argsort(all_fprs)
    sorted_fprs = np.array(all_fprs)[sorted_indices]
    sorted_pros = np.array(all_pros)[sorted_indices]

    return sorted_fprs, sorted_pros



def compute_pro(anomaly_maps:np.ndarray, ground_truth_maps:np.ndarray) -> np.ndarray:
    # Structuring element for computing connected components.
    structure = np.ones((3, 3), dtype=int)

    num_ok_pixels = 0
    num_gt_regions = 0

    shape = (len(anomaly_maps),
             anomaly_maps[0].shape[0],
             anomaly_maps[0].shape[1])
    fp_changes = np.zeros(shape, dtype=np.uint32)
    assert shape[0] * shape[1] * shape[2] < np.iinfo(fp_changes.dtype).max, \
        'Potential overflow when using np.cumsum(), consider using np.uint64.'

    pro_changes = np.zeros(shape, dtype=np.float64)

    for gt_ind, gt_map in enumerate(ground_truth_maps):

        # Compute the connected components in the ground truth map.
        labeled, n_components = label(gt_map, structure)
        num_gt_regions += n_components

        # Compute the mask that gives us all ok pixels.
        ok_mask = labeled == 0
        num_ok_pixels_in_map = np.sum(ok_mask)
        num_ok_pixels += num_ok_pixels_in_map

        # Compute by how much the FPR changes when each anomaly score is
        fp_change = np.zeros_like(gt_map, dtype=fp_changes.dtype)
        fp_change[ok_mask] = 1

        # Compute by how much the PRO changes when each anomaly score is
        pro_change = np.zeros_like(gt_map, dtype=np.float64)
        for k in range(n_components):
            region_mask = labeled == (k + 1)
            region_size = np.sum(region_mask)
            pro_change[region_mask] = 1. / region_size

        fp_changes[gt_ind, :, :] = fp_change
        pro_changes[gt_ind, :, :] = pro_change

    # Flatten the numpy arrays before sorting.
    anomaly_scores_flat = np.array(anomaly_maps).ravel()
    fp_changes_flat = fp_changes.ravel()
    pro_changes_flat = pro_changes.ravel()

    # Sort all anomaly scores.
    sort_idxs = np.argsort(anomaly_scores_flat).astype(np.uint32)[::-1]

    # Info: np.take(a, ind, out=a) followed by b=a instead of
    np.take(anomaly_scores_flat, sort_idxs, out=anomaly_scores_flat)
    anomaly_scores_sorted = anomaly_scores_flat
    np.take(fp_changes_flat, sort_idxs, out=fp_changes_flat)
    fp_changes_sorted = fp_changes_flat
    np.take(pro_changes_flat, sort_idxs, out=pro_changes_flat)
    pro_changes_sorted = pro_changes_flat

    del sort_idxs

    # Get the (FPR, PRO) curve values.
    if num_ok_pixels > 0:
        np.cumsum(fp_changes_sorted, out=fp_changes_sorted)
        fp_changes_sorted = fp_changes_sorted.astype(np.float32, copy=False)
        np.divide(fp_changes_sorted, num_ok_pixels, out=fp_changes_sorted)
        fprs = fp_changes_sorted
    else:
        fprs = np.zeros_like(fp_changes_sorted)

    if num_gt_regions > 0:
        np.cumsum(pro_changes_sorted, out=pro_changes_sorted)
        np.divide(pro_changes_sorted, num_gt_regions, out=pro_changes_sorted)
        pros = pro_changes_sorted
    else:
        pros = np.zeros_like(pro_changes_sorted)


    keep_mask = np.append(np.diff(anomaly_scores_sorted) != 0, np.True_)
    del anomaly_scores_sorted

    fprs = fprs[keep_mask]
    pros = pros[keep_mask]
    del keep_mask

    np.clip(fprs, a_min=None, a_max=1., out=fprs)
    np.clip(pros, a_min=None, a_max=1., out=pros)

    zero = np.array([0.])
    one = np.array([1.])

    return np.concatenate((zero, fprs, one)), np.concatenate((zero, pros, one))