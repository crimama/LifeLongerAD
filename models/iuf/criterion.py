import torch.nn as nn
import torch

from models.iuf.criterion import * 
class IUFCriterion:    
    def __init__(self, config, skip:bool = True):        
        '''
            criterion = IUFCriterion(cfg.criterion)
        '''
        super(IUFCriterion, self).__init__()
        # config에 따라 각 loss criterion 인스턴스 생성 및 속성으로 등록
        for c in config:
            criterion_class = eval(c['type'])
            criterion_instance = criterion_class(**c['kwargs'])
            setattr(self, c['name'], criterion_instance)
            
        self.criterion_list = [c['name'] for c in config]
        
        # SVDLoss의 누적 텐서를 저장하기 위한 버퍼 등록
        # 빈 텐서를 device에 맞게 등록 (CUDA 여부는 추후 device 할당 시 반영됨)
        # self.register_buffer("concatenated_tensor_0", torch.empty(0))
        # self.register_buffer("concatenated_tensor_1", torch.empty(0))
        # self.register_buffer("concatenated_tensor_2", torch.empty(0))
        self.concatenated_tensor_0 = torch.empty(0)
        self.concatenated_tensor_1 = torch.empty(0)
        self.concatenated_tensor_2 = torch.empty(0)

        self.skip = skip 
        
    def __call__(self, outputs: dict, inputs: dict, skip: bool):
        """
        Args:
            outputs (dict): 모델의 forward 결과 (중간 디코더 특징, 분류 결과 등 포함)
            inputs (dict): 입력 데이터 딕셔너리 (예: "clslabel" 포함)
            skip (bool): SVD loss 가중치를 적용할지 여부 결정 플래그

        Returns:
            tuple: (전체 loss, feature_loss, class_loss, svd_loss)
        """
        feature_loss = self._feature_loss(outputs) if 'FeatureMSELoss' in self.criterion_list else 0 

        class_loss = self._ce_loss(outputs, inputs)  if 'CELoss' in self.criterion_list else 0 

        svd_loss = self._svd_loss(outputs) if 'SVDLoss' in self.criterion_list else 0 

        # 전체 loss 계산 (skip 플래그에 따라 SVD 손실 가중치 적용)
        loss = feature_loss + class_loss
        if skip:
            loss += 10 * svd_loss

        return {'loss':loss,
                'feature_loss':feature_loss.item(),
                'class_loss':class_loss.item(),
                'svd_loss':svd_loss.item() 
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
        # 중간 디코더 특징 텐서를 복사 및 누적 (detach로 기울기 전파 차단)
        feat0 = outputs["middle_decoder_feature_0"].clone().detach()
        feat1 = outputs["middle_decoder_feature_1"].clone().detach()
        feat2 = outputs["middle_decoder_feature_2"].clone().detach()
        
        # 모델 출력(feat0)의 device로 빈 텐서를 맞춰줍니다.
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

        # 누적된 텐서 크기가 일정 임계값(예: 768)을 넘으면 앞부분 일부 제거하여 메모리 관리
        if self.concatenated_tensor_0.shape[0] > 768:
            self.concatenated_tensor_0 = self.concatenated_tensor_0[48:]
        if self.concatenated_tensor_1.shape[0] > 768:
            self.concatenated_tensor_1 = self.concatenated_tensor_1[48:]
        if self.concatenated_tensor_2.shape[0] > 768:
            self.concatenated_tensor_2 = self.concatenated_tensor_2[48:]

        # 누적된 텐서를 입력으로 SVD 손실 계산 (SVDLoss는 config에 따라 등록된 criterion)
        return self.SVDLoss(self.concatenated_tensor_0,
                            self.concatenated_tensor_1,
                            self.concatenated_tensor_2)


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