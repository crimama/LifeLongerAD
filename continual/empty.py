import torch
import torch.nn as nn
import torch.nn.functional as F

class EMPTY(nn.Module):
    def __init__(self, model, device='cpu'):
        """
        EWC를 적용하기 위한 Wrapper 클래스
        Args:
            model: 기존 PyTorch 모델
            device: 'cuda' 또는 'cpu'
        """
        super(EMPTY, self).__init__()
        self.model = model
        self.device = device

        self.current_task = 0  # 현재 Task 번호

    def forward(self, *args, **kwargs):
        """기존 모델의 forward를 그대로 호출"""
        return self.model(*args, **kwargs)


    def consolidate(self, trainloader):
        """
        현재 모델의 파라미터를 저장하여 EWC 계산에 사용
        """
        self.current_task += 1

    def criterion(self, outputs, lambda_=1.0):     
        task_loss = self.model.criterion(outputs)
        return task_loss

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
