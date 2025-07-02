import torch.nn as nn
import torch
import torch.nn.functional as F
from models.iuf.criterion import * 

class IUFCriterion:
    def __init__(self, config, skip:bool = True, buffer_size:int = 768):
        super(IUFCriterion, self).__init__()
        self.buffer_size = buffer_size
        self.past_feature_store = {} # 과거 태스크 대표 특징 저장소 초기화
        self.max_stored_tasks = 2  # 최대 저장할 태스크 수
        self.max_features_per_task = 3  # 태스크당 최대 저장할 특징 수
        
        # SVD loss를 위한 텐서 버퍼 초기화
        self.concatenated_tensor_0 = torch.empty(0)
        self.concatenated_tensor_1 = torch.empty(0)
        self.concatenated_tensor_2 = torch.empty(0)

        # config에서 RepresentationOrthogonalityLoss 설정 읽기
        for c in config:
            criterion_class = eval(c['type'])
            if c['type'] == 'RepresentationOrthogonalityLoss':
                # past_feature_store를 인자로 전달
                criterion_instance = criterion_class(weight=c['kwargs'].get('weight', 1.0),
                                                     feature_key_current=c['kwargs'].get('feature_key_current', "middle_decoder_feature_0"), # 예시 키
                                                     past_feature_store=self.past_feature_store,
                                                     eps=c['kwargs'].get('eps', 1e-8))
            else:
                criterion_instance = criterion_class(**c['kwargs'])
            setattr(self, c['name'], criterion_instance)
        self.criterion_list = [c['name'] for c in config]
        # ... 나머지 초기화 ...

    def __call__(self, outputs: dict, cl_manager, current_task_id: int): # current_task_id 추가
        feature_loss = torch.tensor(0.0, device=cl_manager.device)
        svd_loss = torch.tensor(0.0, device=cl_manager.device)
        # ortho_loss = torch.tensor(0.0, device=cl_manager.device) # RepresentationOrthogonalityLoss 내부에서 초기화
        representation_ortho_loss = torch.tensor(0.0, device=cl_manager.device)


        if 'FeatureMSELoss' in self.criterion_list:
            feature_loss = self._feature_loss(outputs)
        if 'SVDLoss' in self.criterion_list:
            svd_loss = self._svd_loss(outputs) * 10
        # if 'OrthoLoss' in self.criterion_list: # 이전 OrthogonalityLoss (파라미터 직교성)
        #     ortho_loss = self.OrthoLoss(cl_manager)
        if 'RepresentationOrthogonalityLoss' in self.criterion_list:
             if self.past_feature_store: # 과거 태스크 정보가 있을 때만 계산
                representation_ortho_loss = self.RepresentationOrthogonalityLoss(outputs, current_task_id)

        loss = feature_loss + svd_loss + representation_ortho_loss # ortho_loss 대신 representation_ortho_loss 사용

        return {
            'loss': loss,
            'feature_loss': feature_loss.item(),
            'svd_loss': svd_loss.item(),
            'representation_ortho_loss': representation_ortho_loss.item()
        }

    def _feature_loss(self, outputs: dict):
        """FeatureMSELoss 기반 손실 계산"""
        return self.FeatureMSELoss(outputs)

    def _ce_loss(self, outputs: dict, inputs: dict):
        """Cross Entropy 손실 계산 (입력 clslabel을 device에 맞게 변환)"""
        cls_label = inputs["clslabel"].to(outputs["class_out"].device)
        return self.CELoss(outputs["class_out"], cls_label)

    def _svd_loss(self, outputs: dict):
        """누적된 중간 디코더 특징을 이용한 SVD 손실 계산
        매 배치마다 새로운 특징 텐서를 누적하고, 일정 크기(여기서는 768 이상) 초과 시 앞부분을 잘라냅니다.
        이후, self.SVDLoss를 호출하여 손실을 계산합니다.
        """
        try:
            # 중간 디코더 특징 텐서를 복사 및 누적 (detach로 기울기 전파 차단)
            feat0 = outputs["middle_decoder_feature_0"].clone().detach()
            feat1 = outputs["middle_decoder_feature_1"].clone().detach()
            feat2 = outputs["middle_decoder_feature_2"].clone().detach()
            
            # 모델 출력(feat0)의 device로 빈 텐서를 맞춰줍니다.
            device = feat0.device
            
            # 기존 텐서들을 device로 이동
            if self.concatenated_tensor_0.device != device:
                self.concatenated_tensor_0 = self.concatenated_tensor_0.to(device)
            if self.concatenated_tensor_1.device != device:
                self.concatenated_tensor_1 = self.concatenated_tensor_1.to(device)
            if self.concatenated_tensor_2.device != device:
                self.concatenated_tensor_2 = self.concatenated_tensor_2.to(device)

            # 새로운 특징 추가 전에 버퍼 크기 제한
            if self.concatenated_tensor_0.shape[0] >= self.buffer_size:
                # 가장 최근의 buffer_size 개수만큼만 유지
                self.concatenated_tensor_0 = self.concatenated_tensor_0[-self.buffer_size:]
                self.concatenated_tensor_1 = self.concatenated_tensor_1[-self.buffer_size:]
                self.concatenated_tensor_2 = self.concatenated_tensor_2[-self.buffer_size:]

            # 새로운 특징 추가
            self.concatenated_tensor_0 = torch.cat([self.concatenated_tensor_0, feat0], dim=0)
            self.concatenated_tensor_1 = torch.cat([self.concatenated_tensor_1, feat1], dim=0)
            self.concatenated_tensor_2 = torch.cat([self.concatenated_tensor_2, feat2], dim=0)

            # 버퍼 크기 제한 확인
            if self.concatenated_tensor_0.shape[0] > self.buffer_size:
                self.concatenated_tensor_0 = self.concatenated_tensor_0[-self.buffer_size:]
                self.concatenated_tensor_1 = self.concatenated_tensor_1[-self.buffer_size:]
                self.concatenated_tensor_2 = self.concatenated_tensor_2[-self.buffer_size:]

            # SVD 손실 계산
            loss = self.SVDLoss(self.concatenated_tensor_0,
                              self.concatenated_tensor_1,
                              self.concatenated_tensor_2)

            # 메모리 정리
            del feat0, feat1, feat2
            torch.cuda.empty_cache()

            return loss

        except Exception as e:
            print(f"Error in _svd_loss: {e}")
            # 에러 발생 시 메모리 정리
            self.concatenated_tensor_0 = torch.empty(0, device=device)
            self.concatenated_tensor_1 = torch.empty(0, device=device)
            self.concatenated_tensor_2 = torch.empty(0, device=device)
            torch.cuda.empty_cache()
            return torch.tensor(0.0, device=device)
    
    def update_past_feature_store(self, task_id: int, representative_features_list: list):
        """
        현재 태스크가 종료된 후, 해당 태스크의 대표 특징들을 저장합니다.
        메모리 관리를 위해 저장되는 태스크 수와 특징 수를 제한합니다.
        
        Args:
            task_id (int): 종료된 태스크의 ID.
            representative_features_list (list): 해당 태스크의 대표 특징 텐서들의 리스트.
        """
        # 1. 저장할 특징 수 제한
        if len(representative_features_list) > self.max_features_per_task:
            # 가장 최근의 특징들만 유지
            representative_features_list = representative_features_list[-self.max_features_per_task:]
        
        # 2. 특징들을 CPU로 이동하고 메모리 효율적으로 저장
        stored_features = []
        for feat in representative_features_list:
            # 특징을 CPU로 이동하고 float32로 변환하여 메모리 사용량 감소
            feat_cpu = feat.clone().detach().cpu().float()
            stored_features.append(feat_cpu)
        
        # 3. 태스크 수 제한
        if len(self.past_feature_store) >= self.max_stored_tasks:
            # 가장 오래된 태스크부터 제거
            oldest_task_id = min(self.past_feature_store.keys())
            del self.past_feature_store[oldest_task_id]
        
        # 4. 새로운 특징 저장
        self.past_feature_store[task_id] = stored_features
        
        # 5. 메모리 정리
        torch.cuda.empty_cache()


class FeatureMSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        feature_rec = input["feature_rec"]
        feature_align = input["feature_align"]
        return self.criterion_mse(feature_rec, feature_align)


class ImageMSELoss(nn.Module):
    """Train a decoder for visualization of reconstructed features"""

    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        image = input["image"]
        image_rec = input["image_rec"]
        return self.criterion_mse(image, image_rec)


class CELoss(nn.CrossEntropyLoss):
    def __init__(self):
        super(CELoss,self).__init__()
        
class SVDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, av0,av1,av2, ratio=0.1):
        
        av0 = av0.mean(dim=2)
        av1 = av1.mean(dim=2)
        av2 = av2.mean(dim=2)
        
        s0 = torch.linalg.svdvals(av0)
        s0 = torch.div(s0, torch.sum(s0))
        cov_loss0 = torch.sum(s0[s0 < ratio/256])
    
        s1 = torch.linalg.svdvals(av1)
        s1 = torch.div(s1, torch.sum(s1))
        cov_loss1 = torch.sum(s1[s1 < ratio/256])

        s2 = torch.linalg.svdvals(av2)
        s2 = torch.div(s2, torch.sum(s2))
        cov_loss2 = torch.sum(s2[s2 < ratio/256])

        return (cov_loss0 + cov_loss1 + cov_loss2)/3

class RepresentationOrthogonalityLoss(nn.Module):
    def __init__(self, weight: float, feature_key_current: str, past_feature_store: dict, eps: float = 1e-8):
        """
        Args:
            weight (float): 이 손실 항의 가중치.
            feature_key_current (str): 현재 태스크의 특징을 outputs 딕셔너리에서 가져올 키.
            past_feature_store (dict): 과거 태스크들의 대표 특징들을 저장하는 딕셔너리.
                                       예: {task_id_0: [feat_tensor1, feat_tensor2, ...],
                                            task_id_1: [feat_tensorA, feat_tensorB, ...]}
            eps (float): 분모가 0이 되는 것을 방지하기 위한 작은 값.
        """
        super().__init__()
        self.weight = weight
        self.feature_key_current = feature_key_current
        self.past_feature_store = past_feature_store # 외부에서 관리 및 업데이트 필요
        self.eps = eps

    def forward(self, outputs: dict, current_task_id: int):
        """
        Args:
            outputs (dict): 모델의 출력을 담고 있는 딕셔너리.
            current_task_id (int): 현재 학습 중인 태스크의 ID.
        """
        current_features = outputs.get(self.feature_key_current)
        if current_features is None:
            return torch.tensor(0.0, device=outputs.get(next(iter(outputs))).device if outputs else "cpu") # outputs가 비어있을 경우 대비

        # 현재 특징을 평균내거나 대표 벡터를 선택 (예시: 배치 평균)
        # (B, C, H, W) 또는 (B, N, D) 형태를 (B, D_flat) 형태로 변환 가정
        if current_features.ndim > 2:
            current_features_flat = current_features.reshape(current_features.size(0), -1)
        else:
            current_features_flat = current_features

        # L2 정규화 (코사인 유사도 계산을 위해)
        current_features_norm = F.normalize(current_features_flat, p=2, dim=1, eps=self.eps)

        ortho_loss = torch.tensor(0.0, device=current_features.device)
        num_comparisons = 0

        for past_task_id, past_features_list in self.past_feature_store.items():
            if past_task_id == current_task_id or not past_features_list: # 현재 태스크 또는 비어있는 리스트는 건너뜀
                continue

            for past_feature_flat in past_features_list:
                if past_feature_flat.ndim > 1: # 이미 flatten된 상태가 아니라면 flatten
                     past_feature_flat_single = past_feature_flat.reshape(1, -1) # 단일 과거 특징으로 간주
                else:
                     past_feature_flat_single = past_feature_flat.unsqueeze(0)

                past_feature_norm = F.normalize(past_feature_flat_single.to(current_features.device), p=2, dim=1, eps=self.eps)

                # 현재 배치의 모든 특징과 과거의 단일 대표 특징 간의 코사인 유사도 제곱을 최소화 (직교성 최대화)
                cosine_similarities_sq = (current_features_norm @ past_feature_norm.T).pow(2)
                ortho_loss += cosine_similarities_sq.sum()
                num_comparisons += current_features_norm.size(0) # 비교 횟수만큼 더함

        if num_comparisons > 0:
            ortho_loss = ortho_loss / num_comparisons

        return ortho_loss * self.weight