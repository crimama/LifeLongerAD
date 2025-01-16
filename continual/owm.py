import torch
import torch.nn as nn
import torch.nn.functional as F

class OWMLoss:
    """
    PyTorch의 Tensor처럼 'backward()'를 호출할 수 있는 '가짜' Loss 객체.
    내부적으로:
      - 실제 loss (Tensor)를 가지고 있음
      - backward()가 호출되면, 먼저 loss.backward() 수행
      - 그 후 OWM Projection을 수행
    """
    def __init__(self, loss_tensor, wrapper, input_data=None):
        """
        Args:
            loss_tensor: 실제 손실(Tensor)
            wrapper: OWMWrapper 인스턴스
            input_data: Projection Matrix 업데이트 및 Gradient 투영에 필요한 입력 데이터
        """
        self.loss_tensor = loss_tensor
        self.wrapper = wrapper
        self.input_data = input_data

    def backward(self, *args, **kwargs):
        # 1) 기본 역전파
        self.loss_tensor.backward(*args, **kwargs)
        # 2) OWM Projection 적용
        self.wrapper._apply_owm_projection(self.input_data)

    def item(self):
        return self.loss_tensor.item()

    def __float__(self):
        return float(self.loss_tensor.item())

    def __str__(self):
        return str(self.loss_tensor)

    def __repr__(self):
        return f"OWMLoss({self.loss_tensor})"


class OWMWrapper(nn.Module):
    def __init__(self, model, input_dim: int, device='cpu'):
        """
        OWM를 적용하기 위한 Wrapper 클래스
        Args:
            model: 기존 PyTorch 모델
            input_dim: 입력 차원 (e.g. MNIST: 784)
            device: 'cuda' 또는 'cpu'
        """
        super().__init__()
        self.model = model
        self.device = device

        # OWM에서 사용하는 Projection Matrix 초기화
        self.P = torch.eye(input_dim, device=self.device)

        # 필요 시 Task 구분
        self.current_task = 0

    def forward(self, x):
        """
        기존 모델의 forward를 그대로 호출
        """
        return self.model(x)

    def criterion(self, inputs, targets=None):
        """
        사용자가 model.criterion(...)으로 Loss를 구하면, 
        실제 Loss(Tensor) 대신 'OWMLoss' 객체를 반환해 
        .backward() 시 OWM Projection이 자동 동작하도록 함.

        Args:
            inputs: 모델 forward에 들어갈 입력
            targets: 레이블(있다면)
        """
        # 1) Forward
        outputs = self.forward(inputs)

        # 2) 모델이 원래 쓰던 Loss 계산 (예: cross_entropy 등)
        base_loss = self.model.criterion(outputs)

        # 3) OWMLoss 객체를 만들어 반환
        return OWMLoss(base_loss, self, input_data=inputs)

    def _apply_owm_projection(self, input_data):
        """
        OWM Projection 적용:
          - (선택) Projection Matrix P 갱신
          - 파라미터의 grad에 P 투영 적용
        """
        if input_data is not None:
            # input_data shape: (batch_size, input_dim)
            if len(input_data.shape) > 2:
                # 예) (batch, 1, 28, 28) -> Flatten
                input_data = input_data.view(input_data.size(0), -1)
            self._update_projection_matrix(input_data)

        # Gradient Projection
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                g_shape = param.grad.shape
                # Flatten grad
                grad_2d = param.grad.view(-1, 1)  # (num_params, 1)
                # OWM Projection
                projected_grad = self.P @ grad_2d
                # 원래 shape로 복원
                param.grad = projected_grad.view(*g_shape)

    def _update_projection_matrix(self, X):
        """
        OWM의 핵심: 입력 X를 이용하여 P(Projection Matrix) 업데이트
        (가장 기본적인 OWM 업데이트 공식을 예시로 표시)
        """
        # X shape: (batch_size, input_dim)
        X_t = X.T  # shape: (input_dim, batch_size)

        # I + X * P * X^T 의 역행렬
        # (batch_size x batch_size)
        middle = torch.inverse(
            torch.eye(X.shape[0], device=self.device) + X @ self.P @ X_t
        )
        self.P = self.P - (self.P @ X_t @ middle @ X @ self.P)

    def consolidate(self):
        """
        Task가 끝날 때 호출 (선택)
        """
        self.current_task += 1

    def __getattr__(self, name):
        """
        Wrapper로 감싼 모델의 모든 메소드에 접근 가능하도록 함
        (예: named_parameters, named_modules 등)
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
