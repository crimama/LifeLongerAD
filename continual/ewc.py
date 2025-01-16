import torch
import torch.nn as nn
import torch.nn.functional as F

class EWC(nn.Module):
    def __init__(self, model):
        """
        EWC를 적용하기 위한 Wrapper 클래스
        Args:
            model: 기존 PyTorch 모델
            device: 'cuda' 또는 'cpu'
        """
        super(EWC, self).__init__()
        self.model = model

        
        # EWC 관련 변수 초기화
        self.params = None  # 이전 Task의 파라미터 저장
        self.fisher = None  # Fisher Matrix 저장
        self.current_task = 0  # 현재 Task 번호

    def forward(self, *args, **kwargs):
        """기존 모델의 forward를 그대로 호출"""
        return self.model(*args, **kwargs)

    def compute_fisher(self, dataloader):
        """
        Fisher Matrix를 계산하는 메소드
        Args:
            dataloader: Fisher 계산을 위한 데이터로더
        """
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()

        for data, target in dataloader:
            data, target = data, target
            self.model.zero_grad()
            output = self.model(data)
            loss = self.model.criterion(output)
            loss.backward()

            # Gradient^2 값을 Fisher에 누적
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2)

        # 평균 계산
        for n in fisher:
            fisher[n] /= len(dataloader.dataset)
        self.fisher = fisher

    def consolidate(self, trainloader):
        """
        현재 모델의 파라미터를 저장하여 EWC 계산에 사용
        """
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self.current_task += 1
        self.compute_fisher(trainloader)

    def compute_ewc_loss(self, lambda_=1.0):
        """
        EWC Loss를 계산하는 메소드
        Args:
            lambda_: EWC 정규화 항의 가중치
        Returns:
            EWC Loss 값
        """
        if self.params is None or self.fisher is None:
            return 0.0  # Task 0인 경우 EWC Loss 없음

        ewc_loss = 0.0
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                fisher_val = self.fisher[n]
                param_val = self.params[n]
                ewc_loss += (fisher_val * (p - param_val).pow(2)).sum()
        return (lambda_ / 2) * ewc_loss

    def criterion(self, outputs, lambda_=1.0):
        """
        Loss 계산 시 EWC Loss 추가
        Args:
            outputs: 모델 출력
            targets: 정답 라벨
            lambda_: EWC 정규화 항의 가중치
        Returns:
            총 Loss 값
        """        
        task_loss = self.model.criterion(outputs)
        
        if self.current_task > 0:  # Task 0 이후에는 EWC Loss 추가
            ewc_loss = self.compute_ewc_loss(lambda_)
            return task_loss + ewc_loss
        
        return task_loss

    def __getattr__(self, name):
        """
        Wrapper로 감싼 모델의 모든 메소드에 접근 가능하도록 함
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
