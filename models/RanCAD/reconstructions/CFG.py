import copy
import math
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
# Assuming initializer and transformer modules are correctly imported from parent directories or installed packages
# from ..initializer import initialize_from_cfg # Example import
from torch import Tensor, nn

# Assuming these are defined elsewhere or imported
from .transformer import Transformer, build_position_embedding # Make sure build_position_embedding is accessible

class CFGReconstruction(nn.Module):
    def __init__(
        self,
        num_classes=15,          # Total number of classes across all tasks eventually
        M=2048,                  # Dimension for Random Projection
        lambda_reg=1e-3,         # Ridge regression regularization parameter
        use_ranpac_classifier=True, # Flag to enable RanPAC classification logic
        **kwargs,
    ):
        super().__init__()
        self.feature_size = kwargs['feature_size']
        self.feature_dim = kwargs['feature_dim']
        self.inplanes = kwargs['inplanes']
        self.cfg_embed_dim = kwargs['cfg_embed_dim'] # Note: This seems unused in the provided forward pass
        self.feature_jitter = kwargs.get('feature_jitter', None) # Use .get for optional args
        self.uncond_prob = kwargs.get('uncond_prob', 0.0) # Use .get for optional args
        self.guidance_scale = kwargs.get('guidance_scale', 1.0) # Use .get for optional args
        self.use_ranpac_classifier = use_ranpac_classifier
        self.num_classes = num_classes # Store total expected classes

        # Existing Projection layers
        self.input_proj = nn.Linear(self.inplanes, self.feature_dim)
        self.rec_head = nn.Linear(self.feature_dim, self.inplanes)

        # --- RanPAC Components ---
        if self.use_ranpac_classifier:
            self.M = M
            self.lambda_reg = lambda_reg

            # 1) RanPAC Random Projection Layer (Frozen)
            # Note: RanPAC applies RP to the features *before* the classifier.
            # Here, we apply it to the output of input_proj (src).
            self.random_proj = nn.Linear(self.feature_dim, M, bias=False)
            nn.init.normal_(self.random_proj.weight, mean=0, std=1/math.sqrt(self.feature_dim)) # Initialize RP matrix
            self.random_proj.weight.requires_grad = False # Freeze the random projection

            # 2) Buffers for Accumulating Statistics for Ridge Regression
            # G: Accumulator for H^T * H (Gram matrix of projected features)
            # C: Accumulator for H^T * Y (Projected features * one-hot labels)
            # W_o: Learned classifier weights (M x num_classes) - can be buffer if updated only here
            self.register_buffer('G', torch.zeros(M, M))
            self.register_buffer('C', torch.zeros(M, self.num_classes))
            self.register_buffer('W_o', torch.zeros(M, self.num_classes))
            # Keep track of classes seen so far to mask output logits if needed
            self.register_buffer('seen_classes_mask', torch.zeros(self.num_classes, dtype=torch.bool))
            self._current_task_idx = -1 # Internal tracker for tasks
            self._known_classes = 0 # Classes known before the current task

        # --- Original Transformer Components ---
        transformer_kwargs = {
             'hidden_dim': kwargs.get('hidden_dim', 256),
             'feature_size': self.feature_size, # Already extracted
             'neighbor_mask': kwargs.get('neighbor_mask', None),
             'nhead': kwargs.get('nhead', 8),
             'num_encoder_layers': kwargs.get('num_encoder_layers', 1),
             'num_decoder_layers': kwargs.get('num_decoder_layers', 4),
             'dim_feedforward': kwargs.get('dim_feedforward', 1024),
             'dropout': kwargs.get('dropout', 0.1),
             'activation': kwargs.get('activation', "relu"),
             'normalize_before': kwargs.get('normalize_before', False),
             'return_intermediate_dec': kwargs.get('return_intermediate_dec', True)
        }
        self.transformer = Transformer(**transformer_kwargs)

        # --- Position Embedding ---
        pos_embed_type = kwargs.get('pos_embed_type', 'learned')
        self.pos_embed = build_position_embedding(pos_embed_type,
                                                  transformer_kwargs['feature_size'],
                                                  transformer_kwargs['hidden_dim'])

        # --- Upsampling ---
        # Ensure 'instrides' is present in kwargs or handle its absence
        if 'instrides' not in kwargs or not kwargs['instrides']:
             raise ValueError("`instrides` is required in kwargs for nn.UpsamplingBilinear2d scale_factor.")
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=kwargs['instrides'][0])


    def update_continual_state(self, current_task_idx, new_class_indices):
        """
        새로운 태스크 학습을 시작하기 *전에* 호출합니다.
        RanPAC 분류기의 내부 상태를 특정 클래스 인덱스를 사용하여 업데이트합니다.

        Args:
            current_task_idx (int): 현재 태스크 인덱스 (추적/로깅용).
            new_class_indices (list or torch.Tensor): 이번 태스크에서 추가되는 새로운 클래스들의
                                                       인덱스를 담고 있는 리스트 또는 텐서.
        """
        self._current_task_idx = current_task_idx

        # 입력된 인덱스를 LongTensor로 변환 (리스트인 경우)
        if isinstance(new_class_indices, list):
            # 빈 리스트 처리: 빈 리스트가 들어오면 LongTensor로 변환 시 에러 발생 가능
            if not new_class_indices:
                 print(f"Task {current_task_idx}: No new class indices provided.")
                 # _known_classes는 변경하지 않음
                 self._known_classes = torch.sum(self.seen_classes_mask).item()
                 print(f"RanPAC state unchanged. Total known classes: {self._known_classes}")
                 return # 추가 작업 없이 종료
            new_class_indices = torch.tensor(new_class_indices, dtype=torch.long)

        # 인덱스 텐서를 마스크와 동일한 device로 이동 (안전을 위해)
        new_class_indices = new_class_indices.to(device=self.seen_classes_mask.device)

        # 인덱스 유효성 검사 (0 미만 또는 num_classes 이상인 경우)
        if torch.any(new_class_indices >= self.num_classes) or torch.any(new_class_indices < 0):
            raise ValueError(f"제공된 클래스 인덱스가 범위를 벗어났습니다 (0 ~ {self.num_classes - 1}). Indices: {new_class_indices.tolist()}")

        # 제공된 인덱스를 사용하여 마스크 업데이트 (Advanced Indexing)
        self.seen_classes_mask[new_class_indices] = True

        # 마스크에서 True인 개수를 세어 known_classes 업데이트
        self._known_classes = torch.sum(self.seen_classes_mask).item()

        print(f"Updated RanPAC state for Task {current_task_idx}. Added indices: {new_class_indices.tolist()}. Total known classes: {self._known_classes}")


    def forward_pre(self, feature_align):
        # Rearrange for Transformer: B C H W -> (H W) B C
        feature_tokens = rearrange(
            feature_align, "b c h w -> (h w) b c"
        )

        # Apply Jitter during training if configured
        if self.training and self.feature_jitter:
            feature_tokens = self.add_jitter(
                feature_tokens, self.feature_jitter['scale'], self.feature_jitter['prob']
            ) # Assuming feature_jitter is a dict {'scale': float, 'prob': float}

        # Project input features
        src = self.input_proj(feature_tokens) # L B C (feature_dim)

        # Calculate position embedding
        pos_embed = self.pos_embed(src) # Requires pos_embed compatible with L B C or similar
        # Ensure pos_embed is L B C if needed by transformer
        # Original code had permute(1,0,2) - check if your transformer needs L, B, C or B, L, C
        # Assuming Transformer needs L, B, C
        # pos_embed = pos_embed.permute(1, 0, 2) if pos_embed.ndim == 3 and pos_embed.shape[0] != src.shape[0] else pos_embed

        return src, pos_embed


    def forward(self, input, targets=None):
        feature_align = input["feature_align"] # B C H W
        L, B, C_feat = -1, feature_align.shape[0], self.feature_dim

        src, pos_embed = self.forward_pre(feature_align)
        L = src.shape[0]

        ranpac_outputs = {}
        if self.use_ranpac_classifier:
            src_flat = rearrange(src, 'l b c -> (l b) c')
            # h_flat는 ranpac_ce_loss 경로에서 그래디언트 필요
            h_flat = F.relu(self.random_proj(src_flat)) # (L*B, M)

            # h_pooled는 로짓 계산에 사용되므로 그래디언트 필요
            h_pooled = h_flat.view(L, B, self.M).mean(dim=0) # (B, M)

            if self.training:
                if targets is None:
                    raise ValueError("Targets (class indices) are required for RanPAC training.")

                # --- W_o 계산 부분을 torch.no_grad()로 감싸기 ---
                with torch.no_grad():
                    # G, C 업데이트를 위한 텐서 준비 (여기서는 detach 불필요, G/C는 버퍼)
                    targets_expanded_no_grad = targets.unsqueeze(0).repeat(L, 1).view(-1)
                    y_onehot_no_grad = F.one_hot(targets_expanded_no_grad, num_classes=self.num_classes).float()

                    # G, C 업데이트 (버퍼에 대한 in-place 연산)
                    # h_flat_detached = h_flat.detach() # G,C 업데이트에 detach된 버전 사용? -> 필요 없을 가능성 높음
                    # self.G += h_flat_detached.T @ h_flat_detached
                    # self.C += h_flat_detached.T @ y_onehot_no_grad
                    # 만약 위 detach 없이도 에러나면 아래처럼 h_flat을 직접 사용
                    self.G += h_flat.T @ h_flat
                    self.C += h_flat.T @ y_onehot_no_grad


                    # 업데이트된 G, C를 사용하여 W_o 계산
                    I = torch.eye(self.M, device=self.G.device)
                    try:
                        # 계산 결과를 W_o 버퍼에 직접 할당 (in-place update와 유사)
                        self.W_o[...] = torch.linalg.solve(self.G + self.lambda_reg * I, self.C)
                    except torch.linalg.LinAlgError:
                        print("Warning: G matrix potentially singular. Adding epsilon.")
                        epsilon = 1e-6
                        self.W_o[...] = torch.linalg.solve(self.G + self.lambda_reg * I + epsilon * I, self.C)
                # --- torch.no_grad() 끝 ---

                # 이제 업데이트된 self.W_o를 사용하여 로짓 계산
                # W_o는 이 역전파 단계에서는 상수로 취급됨
                # h_pooled는 여전히 src로부터 그래디언트를 받음
                logits = h_pooled @ self.W_o
                logits[:, ~self.seen_classes_mask.to(logits.device)] = -float('inf')
                ranpac_outputs["ranpac_logits"] = logits

            else: # Inference
                # 추론 시에는 G/C/W_o 업데이트 없음
                logits = h_pooled @ self.W_o
                logits[:, ~self.seen_classes_mask.to(logits.device)] = -float('inf')
                ranpac_outputs["ranpac_logits"] = logits

        # --- Original Reconstruction Logic ---
        transformer_input = src
        output_decoder, memory = self.transformer(transformer_input, pos_embed)

        # ... (중간 특징 처리, 최종 재구성, 이상 점수 계산 로직은 동일) ...
        middle_features = {}
        if output_decoder.ndim == 4:
            # ... (4D 처리) ...
             num_layers, L_out, B_out, C_out = output_decoder.shape
             final_output_decoder = output_decoder[-1, :, :, :]
             for i in range(min(3, num_layers - 1)):
                 mid_feat = output_decoder[i, :, :, :]
                 mid_feat_rec = rearrange(mid_feat, "(h w) b c -> b c h w", h=self.feature_size[0], w=self.feature_size[1])
                 middle_features[f"middle_decoder_feature_{i}"] = mid_feat_rec
        elif output_decoder.ndim == 3:
             final_output_decoder = output_decoder
        else:
             raise ValueError(f"Unexpected shape for output_decoder: {output_decoder.shape}")

        rec_feature_tokens = self.rec_head(final_output_decoder)
        feature_rec = rearrange( rec_feature_tokens, "(h w) b c -> b c h w", h=self.feature_size[0], w=self.feature_size[1])
        pred_diff_sq = (feature_rec - feature_align) ** 2
        pred_l2 = torch.sqrt(torch.sum(pred_diff_sq, dim=1, keepdim=True))
        pred = self.upsample(pred_l2)


        # --- Combine Outputs ---
        output_dict = {
            "feature_rec": feature_rec,
            "feature_align": feature_align,
            "pred": pred,
        }
        output_dict.update(middle_features)
        output_dict.update(ranpac_outputs)

        return output_dict

    # ... (add_jitter 함수 동일) ...
    def add_jitter(self, feature_tokens, scale, prob):
        # Original jitter function
        if random.uniform(0, 1) <= prob:
            num_tokens, batch_size, dim_channel = feature_tokens.shape
            # Normalize scale by sqrt(dim_channel) for stability? RanPAC paper doesn't specify this.
            # feature_norms = (
            #     feature_tokens.norm(dim=2, keepdim=True) / math.sqrt(dim_channel)
            # )
            feature_norms = feature_tokens.norm(dim=2, keepdim=True) # L B 1
            jitter = torch.randn_like(feature_tokens) # Use randn_like for correct shape and device
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens