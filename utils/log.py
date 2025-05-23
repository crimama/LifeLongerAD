import logging
import logging.handlers
import os 
import json 
import wandb 
import numpy as np
from collections import OrderedDict
import torch
from river.drift import ADWIN
from logging.handlers import RotatingFileHandler
    
    

class DriftMonitor:
    def __init__(self, log_dir: str = None, log_path: str = None, logger_name: str = "DriftMonitor.log"):
        """
        ADWIN을 이용한 드리프트 모니터링 클래스.
        
        Args:
            log_dir (str): 로그 파일을 저장할 디렉토리. log_path가 None일 때 사용.
            log_path (str): 로그 파일 전체 경로. 지정되면 RotatingFileHandler를 사용.
            logger_name (str): logger의 이름.
        """
        self.adwin = ADWIN()
        self.window_means = []
        self.window_widths = []
        self.drift_magnitudes = []
        self.prev_window_mean = None
        self.step = 0

        # Logger 생성
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # 상위 로거로 전달 방지
        self.logger.handlers.clear()  # 기존 핸들러 제거

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 터미널 출력 제거 (StreamHandler를 추가하지 않음)

        # log_path가 지정되면 RotatingFileHandler 사용
        if log_path:
            log_dir_path = os.path.dirname(log_path)
            if log_dir_path and not os.path.exists(log_dir_path):
                os.makedirs(log_dir_path, exist_ok=True)
            file_handler = RotatingFileHandler(log_path, maxBytes=2 * (1024 ** 2), backupCount=3)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        elif log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            file_path = os.path.join(log_dir, f"{logger_name}")
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def update(self, features: torch.Tensor):
        """
        입력된 features(예: latent representation)를 이용하여 L2 norm을 계산한 후, ADWIN에 업데이트하고,
        내부 통계를 기록합니다.
        
        Args:
            features (torch.Tensor): 모델의 latent representation.
            
        Returns:
            detected (bool): 드리프트가 감지되었으면 True.
            current_mean (float): 현재 ADWIN 윈도우의 평균.
            drift_magnitude (float): 이전 평균과의 차이 (드리프트 강도).
        """
        latent = features.detach().cpu().numpy()
        norm_value = np.linalg.norm(latent.reshape(latent.shape[0], -1), axis=1)
        
        for norm in norm_value:
            self.adwin.update(norm)
            current_mean = self.adwin.estimation
            current_width = self.adwin.width

            self.window_means.append(current_mean)
            self.window_widths.append(current_width)

            if self.prev_window_mean is not None:
                drift_magnitude = abs(current_mean - self.prev_window_mean)
            else:
                drift_magnitude = 0.0
            self.drift_magnitudes.append(drift_magnitude)

            self.prev_window_mean = current_mean
            self.step += 1

            # 로그 기록 (터미널 출력 없이 파일에만 기록됨)
            self.logger.info(f"Step {self.step}: current={norm:.32f}, width={current_width}, magnitude={drift_magnitude:.4f}")

class FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt='%(levelname)s: %(message)s'):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)


def setup_default_logging(default_level=logging.INFO, log_path=''):
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(FormatterNoInfo())
    logging.root.addHandler(console_handler)
    logging.root.setLevel(default_level)
    if log_path:
        with open(log_path, 'a') as f: 
            f.write('\n')
        file_handler = logging.handlers.RotatingFileHandler(log_path, maxBytes=(1024 ** 2 * 2), backupCount=3)
        file_formatter = logging.Formatter("%(asctime)s - %(name)20s: [%(levelname)8s] - %(message)s")
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)
        
        
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
                          
def metric_logging(savedir, use_wandb=None,
                    epoch_time_m=None, epoch = None,
                    optimizer=None,test_metrics=None,task=0,train_metrics=None,
                    class_name=None, current_class_name=None, **kwargs):
    
    metrics = OrderedDict()
    metrics.update([('task',task)])
    
    if epoch is not None:
        metrics.update([('epoch', epoch)])
    
    if optimizer is not None:
        metrics.update([('lr',round(optimizer.param_groups[0]['lr'],5))])
        
    if train_metrics is not None:
        metrics.update([('train_metrics', train_metrics)])
        
    if class_name is not None:
        metrics.update([('class_name', class_name)])
        
    if current_class_name is not None:
        metrics.update([('GT_class_name', current_class_name)])
        
    metrics.update([
                    ('test_metrics',test_metrics)
                    ])
    if epoch_time_m is not None:
        metrics.update([('epoch_time',round(epoch_time_m.val,4))])
        
    # 추가적인 인자를 kwargs로 처리하여 metrics에 포함
    for key, value in kwargs.items():
        if isinstance(value, (int, float)):
            metrics.update([(key, round(value, 4))])
        else:
            metrics.update([(key, value)])
    
    # 결과 저장
    save_metrics_to_csv(dict(metrics), os.path.join(savedir,'results/result_log.csv'))
    
    # wandb 로깅
    if use_wandb and test_metrics is not None:
        # Helper function to convert ListConfig to regular list
        def convert_listconfig(obj):
            if hasattr(obj, '__class__') and obj.__class__.__name__ == 'ListConfig':
                return list(obj)
            elif isinstance(obj, (list, tuple)):
                return list(obj)
            return obj
            
        wandb_metrics = {
            'Train/Epoch': epoch,
            'Test/class_name': convert_listconfig(class_name),
            'Test/GT_class_name': convert_listconfig(current_class_name),
            'Time/Test Batch Average (s)': epoch_time_m.avg if epoch_time_m is not None else 0
        }
        
        # img_level 메트릭 동적 추가
        if 'img_level' in test_metrics:
            for metric_name, metric_value in test_metrics['img_level'].items():
                wandb_metrics[f'Test/img_level_{metric_name}'] = convert_listconfig(metric_value)
        
        # pix_level 메트릭 동적 추가
        if 'pix_level' in test_metrics:
            for metric_name, metric_value in test_metrics['pix_level'].items():
                wandb_metrics[f'Test/pix_level_{metric_name}'] = convert_listconfig(metric_value)
        
        # train_metrics가 있으면 추가
        if train_metrics is not None:
            for metric_name, metric_value in train_metrics.items():
                wandb_metrics[f'Train/{metric_name}'] = convert_listconfig(metric_value)
        
        wandb.log(wandb_metrics)
        
        
def save_metrics_to_csv(metrics, csv_filename):
    import csv 
    # 기본 메타데이터 필드 
    flat_data = {
        "task": metrics.get('task', 0),
        "epoch": metrics.get('epoch', 0),
        "class_name": metrics.get('class_name', ''),
        "GT_class_name": metrics.get('GT_class_name', ''),
        "epoch_time": metrics.get('epoch_time', 0),
        "last": metrics.get('last', False)
    }
    
    # 동적으로 test_metrics의 img_level과 pix_level 메트릭 추가
    if 'test_metrics' in metrics:
        test_metrics = metrics['test_metrics']
        
        # img_level 메트릭 추가
        if 'img_level' in test_metrics:
            for metric_name, metric_value in test_metrics['img_level'].items():
                flat_data[f"img_level_{metric_name}"] = metric_value
                
        # pix_level 메트릭 추가
        if 'pix_level' in test_metrics:
            for metric_name, metric_value in test_metrics['pix_level'].items():
                flat_data[f"pix_level_{metric_name}"] = metric_value
    
    # train_metrics가 있는 경우 추가
    if 'train_metrics' in metrics:
        for metric_name, metric_value in metrics['train_metrics'].items():
            flat_data[f"train_{metric_name}"] = metric_value

    # 파일이 존재하는지 확인해서 존재하지 않으면 헤더를 추가
    file_exists = os.path.isfile(csv_filename)
    
    # 파일 경로의 디렉토리가 존재하지 않으면 생성
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    # CSV 파일에 추가 모드로 데이터 저장
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=flat_data.keys())
        
        # 파일이 처음 생성될 때는 헤더를 작성
        if not file_exists:
            writer.writeheader()

        # 데이터 행 추가
        writer.writerow(flat_data)
        

def wandb_init(config):
    log_config = extract_experiment_summary(config)
    wandb.init(name=config.DEFAULT.exp_name, project=config.TRAIN.wandb.project_name, config=log_config, group=config.DEFAULT.exp_name.split('-')[1])   
        
def extract_experiment_summary(config):
    """
    YAML 설정 딕셔너리에서 실험의 핵심 요소만 추출하여 새로운 딕셔너리로 반환합니다.
    """
    summary = {}

    # --- 기본 설정 ---
    default_cfg = config.get('DEFAULT', {})
    summary['seed'] = default_cfg.get('seed')
    summary['exp_name'] = default_cfg.get('exp_name') # 실험 이름 포함

    # --- 데이터셋 설정 ---
    dataset_cfg = config.get('DATASET', {})
    summary['dataset_name'] = dataset_cfg.get('dataset_name')
    summary['batch_size'] = dataset_cfg.get('batch_size')
    summary['img_size'] = dataset_cfg.get('img_size')
    
    # 데이터셋 파라미터 (Anomaly Ratio, Baseline 등)
    dataset_params = dataset_cfg.get('params', {})
    summary['anomaly_ratio'] = dataset_params.get('anomaly_ratio')
    summary['baseline_mode'] = dataset_params.get('baseline')

    # 클래스 이름 목록 단순화 (중첩 리스트 -> 단일 리스트)
    original_class_names = dataset_cfg.get('class_names', [])
    simplified_class_names = []
    if isinstance(original_class_names, list):
        for item in original_class_names:
            if isinstance(item, list):
                simplified_class_names.extend(item)
            else:
                simplified_class_names.append(item)
    summary['dataset_class_names'] = simplified_class_names

    # --- 옵티마이저 설정 ---
    optimizer_cfg = config.get('OPTIMIZER', {})
    summary['optimizer_name'] = optimizer_cfg.get('opt_name')
    summary['learning_rate'] = optimizer_cfg.get('lr')
    # 옵티마이저 상세 파라미터 (betas 등)
    summary['optimizer_params'] = optimizer_cfg.get('params') 

    # --- 학습 설정 ---
    train_cfg = config.get('TRAIN', {})
    summary['epochs'] = train_cfg.get('epochs')
    summary['grad_accum_steps'] = train_cfg.get('grad_accum_steps')
    summary['mixed_precision'] = train_cfg.get('mixed_precision')

    # --- 연속 학습 설정 ---
    continual_cfg = config.get('CONTINUAL', {})
    summary['is_continual'] = continual_cfg.get('continual')
    summary['is_online'] = continual_cfg.get('online')
    # 연속 학습 방법 이름 (EMPTY가 아니라면 중요할 수 있음)
    summary['continual_method'] = continual_cfg.get('method', {}).get('name') 

    # --- 스케줄러 설정 ---
    scheduler_cfg = config.get('SCHEDULER', {})
    scheduler_name = scheduler_cfg.get('name')
    # 스케줄러가 'null'이나 None이 아닐 경우에만 이름과 파라미터 포함
    if scheduler_name and scheduler_name.lower() != 'null':
        summary['scheduler_name'] = scheduler_name
        summary['scheduler_params'] = scheduler_cfg.get('params')
    else:
        summary['scheduler_name'] = None # 명시적으로 스케줄러 없음 표시
        
    # --- 모델 파라미터 
    summary.update(config.MODEL.params.net_cfg[2].kwargs)
    return summary