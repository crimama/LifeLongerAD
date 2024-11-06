import logging
import logging.handlers
import os 
import json 
import wandb 
from collections import OrderedDict

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
    
    save_metrics_to_csv(dict(metrics), os.path.join(savedir,'result.csv'))
    
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
        "task agnostic" : metrics['task_agnostic']
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