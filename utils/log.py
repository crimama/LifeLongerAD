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
            self.logger.info(f"Step {self.step}: current_mean={current_mean:.4f}, width={current_width}, magnitude={drift_magnitude:.4f}")

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
        metrics.update([('train_' + k, round(v,4)) for k, v in train_metrics.items()])
        
    if class_name is not None:
        metrics.update([('class_name', class_name)])
        
    if current_class_name is not None:
        metrics.update([('GT_class_name', current_class_name)])
        
    metrics.update([
                    # ('test_' + k, round(v,4)) for k, v in test_metrics.items()
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
    
    # with open(os.path.join(savedir, 'result.txt'),  'a') as f:
    #     f.write(json.dumps(metrics) + "\n")
    
    save_metrics_to_csv(dict(metrics), os.path.join(savedir,'results/result_log.csv'))
    
    if use_wandb:
        wandb.log(metrics, step=epoch)
        
        
def save_metrics_to_csv(metrics, csv_filename):
    import csv 
    # 딕셔너리 데이터를 평평하게 변환
    flat_data = {
        "task": metrics['task'],
        "epoch": metrics['epoch'],
        "class_name": metrics['class_name'],
        "GT_class_name": metrics['GT_class_name'],
        "img_level_auroc": metrics['test_metrics']['img_level']['auroc'],
        "img_level_average_precision": metrics['test_metrics']['img_level']['average_precision'],
        "pix_level_auroc": metrics['test_metrics']['pix_level']['auroc'],
        "pix_level_average_precision": metrics['test_metrics']['pix_level']['average_precision'],
        "epoch_time": metrics['epoch_time'],
        "last" : metrics['last']
    }

    # 파일이 존재하는지 확인해서 존재하지 않으면 헤더를 추가
    file_exists = os.path.isfile(csv_filename)

    # CSV 파일에 추가 모드로 데이터 저장
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=flat_data.keys())
        
        # 파일이 처음 생성될 때는 헤더를 작성
        if not file_exists:
            writer.writeheader()

        # 데이터 행 추가
        writer.writerow(flat_data)