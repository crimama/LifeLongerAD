import torch.nn as nn
import torch

# models.iuf.criterion 내에 정의된 loss 클래스들을 import한다고 가정
from models.iuf.criterion import * # 예시, 실제 경로에 맞게 수정 필요

class IUFCriterion:
    def __init__(self, config, skip:bool = True):
        '''
        Args:
            config (list): 각 criterion 설정 딕셔너리의 리스트.
                           예: [{'type': 'FeatureMSELoss', 'name': 'FeatureMSELoss', 'kwargs': {'weight': 1.0}},
                                {'type': 'SVDLoss', 'name': 'SVDLoss', 'kwargs': {}},
                                {'type': 'CELoss', 'name': 'RanPACCELoss', 'kwargs': {'weight': 0.1}}] # RanPAC CE Loss 추가
            skip (bool): SVD loss 가중치 적용 여부 (현재 코드에서는 사용되지 않음)
        '''
        super(IUFCriterion, self).__init__()
        self.criterion_config = config # 나중에 가중치 참조를 위해 저장

        # config에 따라 각 loss criterion 인스턴스 생성 및 속성으로 등록
        for c in config:
            criterion_class = eval(c['type']) # eval 사용에 주의, 안전한 대안 고려 가능
            criterion_instance = criterion_class(**c['kwargs'])
            setattr(self, c['name'], criterion_instance)

        self.criterion_list = [c['name'] for c in config]

        # SVDLoss 누적용 버퍼
        self.concatenated_tensor_0 = torch.empty(0)
        self.concatenated_tensor_1 = torch.empty(0)
        self.concatenated_tensor_2 = torch.empty(0)

        self.skip = skip # skip 플래그는 현재 __call__에서 사용되지 않음

    def _get_criterion_weight(self, name):
        """설정에서 criterion의 가중치를 가져옵니다."""
        for c in self.criterion_config:
            if c['name'] == name:
                # kwargs에 'weight'가 있으면 해당 값을, 없으면 1.0을 기본값으로 반환
                return c['kwargs'].get('weight', 1.0)
        return 1.0 # 해당 이름의 criterion이 없으면 기본 가중치 1.0 반환

    def __call__(self, outputs: dict, inputs: dict):
        """
        Args:
            outputs (dict): 모델 forward 결과 (예: "feature_rec", "feature_align", "middle_decoder_feature_*", "ranpac_logits")
            inputs (dict): 입력 데이터 딕셔너리 (예: "clslabel")
            skip (bool): 현재 __call__ 내에서는 사용되지 않음. SVD 가중치 조절 등에 사용 가능.

        Returns:
            dict: {'loss': total_loss, 'feature_loss': ..., 'svd_loss': ..., 'ranpac_ce_loss': ...}
        """
        total_loss = 0
        loss_dict = {}

        # 1. Feature Reconstruction Loss (MSE)
        if 'FeatureMSELoss' in self.criterion_list:
            feature_loss = self._feature_loss(outputs)
            weight = self._get_criterion_weight('FeatureMSELoss')
            total_loss += feature_loss * weight
            loss_dict['feature_loss'] = feature_loss.item()
        else:
             loss_dict['feature_loss'] = 0

        # 2. SVD Loss
        if 'SVDLoss' in self.criterion_list:
            # SVD Loss는 중간 특징들이 필요함
            if all(f"middle_decoder_feature_{i}" in outputs for i in range(3)):
                svd_loss = self._svd_loss(outputs)
                weight = self._get_criterion_weight('SVDLoss')
                total_loss += svd_loss * weight # SVD loss 가중치 적용
                loss_dict['svd_loss'] = svd_loss.item()
            else:
                # print("Warning: Missing middle decoder features for SVDLoss.")
                loss_dict['svd_loss'] = 0
        else:
             loss_dict['svd_loss'] = 0

        # 3. RanPAC Classification Loss (Cross Entropy) - 보조 손실
        # 'RanPACCELoss'는 config에서 CELoss를 사용하여 정의했다고 가정
        if 'RanPACCELoss' in self.criterion_list:
            if "ranpac_logits" in outputs and "clslabel" in inputs:                
                ranpac_ce_loss = self._ce_loss(outputs, inputs, criterion_name='RanPACCELoss')
                weight = self._get_criterion_weight('RanPACCELoss') # 설정된 가중치 가져오기
                total_loss += ranpac_ce_loss * weight # 가중치 적용
                loss_dict['ranpac_ce_loss'] = ranpac_ce_loss.item()
            else:
                # print("Warning: Missing 'ranpac_logits' in outputs or 'clslabel' in inputs for RanPACCELoss.")
                loss_dict['ranpac_ce_loss'] = 0.0
        else:
            loss_dict['ranpac_ce_loss'] = 0.0


        loss_dict['loss'] = total_loss # 최종 가중합된 손실

        return loss_dict

    def _feature_loss(self, outputs: dict):
        """FeatureMSELoss 계산"""
        # config에서 name='FeatureMSELoss'로 등록된 인스턴스 사용
        return self.FeatureMSELoss(outputs)

    def _ce_loss(self, outputs: dict, inputs: dict, criterion_name: str):
        """Cross Entropy 손실 계산"""
        # criterion_name으로 등록된 CELoss 인스턴스 사용
        criterion_instance = getattr(self, criterion_name)
        # 모델 출력과 입력에서 필요한 텐서 추출
        logits = outputs["ranpac_logits"] # 모델 forward에서 반환된 로짓 사용
        cls_label = inputs["clslabel"].to(logits.device) # 로짓과 같은 device로 이동
        return criterion_instance(logits, cls_label)

    def _svd_loss(self, outputs: dict):
        """누적된 중간 디코더 특징 이용한 SVD 손실 계산"""
        # 중간 디코더 특징 텐서 복사 및 누적
        feat0 = outputs["middle_decoder_feature_0"].clone().detach()
        feat1 = outputs["middle_decoder_feature_1"].clone().detach()
        feat2 = outputs["middle_decoder_feature_2"].clone().detach()

        # Device 맞추기 및 누적
        device = feat0.device
        if self.concatenated_tensor_0.device != device:
            self.concatenated_tensor_0 = self.concatenated_tensor_0.to(device)
        if self.concatenated_tensor_1.device != device:
            self.concatenated_tensor_1 = self.concatenated_tensor_1.to(device)
        if self.concatenated_tensor_2.device != device:
            self.concatenated_tensor_2 = self.concatenated_tensor_2.to(device)

        self.concatenated_tensor_0 = torch.cat([self.concatenated_tensor_0, feat0], dim=0)
        self.concatenated_tensor_1 = torch.cat([self.concatenated_tensor_1, feat1], dim=0)
        self.concatenated_tensor_2 = torch.cat([self.concatenated_tensor_2, feat2], dim=0)

        # 메모리 관리 (임계값 초과 시 오래된 데이터 제거)
        max_len = 768 # 예시 임계값
        trim_len = 48  # 예시 제거 길이
        if self.concatenated_tensor_0.shape[0] > max_len:
            self.concatenated_tensor_0 = self.concatenated_tensor_0[trim_len:]
        if self.concatenated_tensor_1.shape[0] > max_len:
            self.concatenated_tensor_1 = self.concatenated_tensor_1[trim_len:]
        if self.concatenated_tensor_2.shape[0] > max_len:
            self.concatenated_tensor_2 = self.concatenated_tensor_2[trim_len:]

        # 누적 텐서로 SVD 손실 계산
        # config에서 name='SVDLoss'로 등록된 인스턴스 사용
        return self.SVDLoss(self.concatenated_tensor_0,
                            self.concatenated_tensor_1,
                            self.concatenated_tensor_2)

# --- 아래 Loss 클래스 정의는 기존과 동일 ---
class FeatureMSELoss(nn.Module):
    def __init__(self, weight=1.0): # weight 인자 추가 (config에서 받을 수 있도록)
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight # 가중치는 __call__ 에서 적용되므로 여기서는 저장만

    def forward(self, input):
        feature_rec = input["feature_rec"]
        feature_align = input["feature_align"]
        return self.criterion_mse(feature_rec, feature_align)

# ... (ImageMSELoss 정의, 필요하다면 weight 추가) ...

class CELoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs): # weight 인자 추가 및 다른 CE 관련 인자 받을 수 있도록
        super().__init__(**kwargs) # CrossEntropyLoss의 원래 인자들(label_smoothing 등) 전
        
class SVDLoss(nn.Module):
    def __init__(self, weight=1.0, ratio=0.1): # weight, ratio 인자 추가
        super().__init__()
        self.weight = weight
        self.ratio = ratio # SVD 계산 시 사용할 ratio

    def forward(self, av0, av1, av2): # ratio 인자 제거 (init에서 받음)
        # ... (SVD 계산 로직 동일) ...
        av0 = av0.mean(dim=2)
        av1 = av1.mean(dim=2)
        av2 = av2.mean(dim=2)

        s0 = torch.linalg.svdvals(av0)
        s0 = torch.div(s0, torch.sum(s0))
        cov_loss0 = torch.sum(s0[s0 < self.ratio / 256]) # self.ratio 사용

        s1 = torch.linalg.svdvals(av1)
        s1 = torch.div(s1, torch.sum(s1))
        cov_loss1 = torch.sum(s1[s1 < self.ratio / 256]) # self.ratio 사용

        s2 = torch.linalg.svdvals(av2)
        s2 = torch.div(s2, torch.sum(s2))
        cov_loss2 = torch.sum(s2[s2 < self.ratio / 256]) # self.ratio 사용

        return (cov_loss0 + cov_loss1 + cov_loss2) / 3