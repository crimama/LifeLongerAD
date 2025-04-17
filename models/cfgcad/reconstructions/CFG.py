import copy
import math
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from ..initializer import initialize_from_cfg
from torch import Tensor, nn

from PIL import Image
from .transformer import * 



class CFGReconstruction(nn.Module):
    def __init__(
        self,
        num_classes=15,        
        **kwargs,
    ):
        super().__init__()
        self.feature_size = kwargs['feature_size']
        self.feature_dim = kwargs['feature_dim']
        self.inplanes = kwargs['inplanes']
        self.num_classes = num_classes
        self.cfg_embed_dim = kwargs['cfg_embed_dim']
        self.feature_jitter = kwargs['feature_jitter']
        self.uncond_prob = kwargs['uncond_prob']
        self.guidance_scale = kwargs['guidance_scale']
        

        # Task Conditioning Embedding
        self.task_embedding = nn.Embedding(self.num_classes + 1, self.cfg_embed_dim)
        self.null_token_idx = self.num_classes

        # Projection layer
        self.task_proj = nn.Linear(self.cfg_embed_dim, self.feature_dim)
        self.input_proj = nn.Linear(self.inplanes, self.feature_dim)
        self.rec_head = nn.Linear(self.feature_dim, self.inplanes)

        # 설정 파일의 kwargs를 사용하여 Transformer 초기화
        transformer_kwargs = {
             'hidden_dim': kwargs.get('hidden_dim', 256),
             'feature_size': kwargs.get('feature_size'), # feature_size 계산 로직 필요 (아래 참고)
             'neighbor_mask': kwargs.get('neighbor_mask', None), # neighbor_mask 객체 전달 방식 확인
             'nhead': kwargs.get('nhead', 8),
             'num_encoder_layers': kwargs.get('num_encoder_layers', 4),
             'num_decoder_layers': kwargs.get('num_decoder_layers', 4),
             'dim_feedforward': kwargs.get('dim_feedforward', 1024),
             'dropout': kwargs.get('dropout', 0.1),
             'activation': kwargs.get('activation', "relu"),
             'normalize_before': kwargs.get('normalize_before', False),
             'return_intermediate_dec': kwargs.get('return_intermediate_dec', True) # IUF 기본값 사용
        }
        
        # feature_size 는 H', W' 이므로 계산 필요 (예: num_patches 기반)
        if 'num_patches' in kwargs:
             grid_size = int(kwargs['num_patches']**0.5)
             transformer_kwargs['feature_size'] = (grid_size, grid_size)
        elif 'feature_size' not in transformer_kwargs or transformer_kwargs['feature_size'] is None:
             raise ValueError("Transformer `feature_size` (or `num_patches` in kwargs) is required.")

        # neighbor_mask 객체 처리 (설정 파일에서 객체 직접 생성 어려움)
        # -> neighbor_mask 관련 설정값(neighbor_size, mask)만 받고 내부에서 객체 생성 또는 None 처리
        if 'neighbor_mask' in kwargs and kwargs['neighbor_mask'] is not None:
            transformer_kwargs['neighbor_mask'] = kwargs['neighbor_mask']
        else:
             transformer_kwargs['neighbor_mask'] = None # 명시적으로 None 설정

        # --- Transformer 인스턴스 생성 (OASA 없는 버전) ---
        self.transformer = Transformer(**transformer_kwargs)

        # --- 위치 임베딩 생성 (learned 또는 sine) ---
        pos_embed_type = kwargs.get('pos_embed_type', 'learned')
        # build_position_embedding 함수가 이 파일 또는 import 가능해야 함
        self.pos_embed = build_position_embedding(pos_embed_type,
                                                 transformer_kwargs['feature_size'],
                                                 transformer_kwargs['hidden_dim'])

        

        # Initializer (필요시 IUF의 initializer.py 사용)
        initializer_cfg = kwargs.get('initializer')
        if initializer_cfg:
             # from ..initializer import initialize_from_cfg # 예시
             # initialize_from_cfg(self, initializer_cfg)
             pass # 초기화 로직 추가

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=kwargs['instrides'][0])

    def forward_pre(self, feature_align):
        feature_tokens = rearrange(
            feature_align, "b c h w -> (h w) b c"
        )  # --- 입력 shape 변경: B, L, C -> L, B, C --- Transformer는 (Sequence Length, Batch, Channel) 형태를 기대함
        
        ##! ADD Noise 
        if self.training and self.feature_jitter:
            feature_tokens = self.add_jitter(
                feature_tokens, self.feature_jitter.scale, self.feature_jitter.prob
            )            
        
        pos_embed = self.pos_embed(feature_tokens) # PositionEmbedding 모듈의 forward 호출
        pos_embed = pos_embed.permute(1, 0, 2) if pos_embed.ndim == 3 else pos_embed # L, B, C 또는 L, C -> L=H*W                      
        
        src = self.input_proj(feature_tokens)
        return src, pos_embed
        
        
    def forward(self, input, task_id=None):
        feature_align = input["feature_align"]  # B x C X H x W #? MFCN에서 size 맞춰준 feature
        src, pos_embed = self.forward_pre(feature_align)
        device = feature_align.device
        if self.training:
            if task_id is None: raise ValueError("task_id required for training")
            B = feature_align.shape[0]
            
            uncond_mask = torch.rand(B, device=device) < self.uncond_prob
            masked_task_id = task_id.clone()
            masked_task_id[uncond_mask] = self.null_token_idx

            task_emb = self.task_embedding(masked_task_id) # B, D_cfg
            task_emb_proj = self.task_proj(task_emb) # B, D_feat

            #! --- Task 임베딩을 src에 더하기 ---
            # src: L, B, C / task_emb_proj: B, C -> unsqueeze -> 1, B, C
            # conditioned_src = src + task_emb_proj.unsqueeze(0)
            conditioned_src = src
                                    
            #! --- Single pass through Transformer ---
            output_decoder, _ = self.transformer(conditioned_src, pos_embed, task_emb_proj.unsqueeze(0)) # mask 인자 필요시 추가
            middle_decoder_feature=output_decoder[0:3,...]
               
            middle_decoder_feature_rec_0 = rearrange(
                middle_decoder_feature[0], "(h w) b c -> b c (h w)", h=self.feature_size[0]
            )  
            middle_decoder_feature_rec_1 = rearrange(
                middle_decoder_feature[1], "(h w) b c -> b c (h w)", h=self.feature_size[0]
            )  
            middle_decoder_feature_rec_2 = rearrange(
                middle_decoder_feature[2], "(h w) b c -> b c (h w)", h=self.feature_size[0]
            )   
            
            ##! 최종 reconstruction 출력 
            output_decoder = output_decoder[3]
            # rec_tokens = output_decoder.permute(1, 0, 2) # --- 출력 shape 변경: L, B, C -> B, L, C ---            
            rec_feature = self.rec_head(output_decoder)
            feature_rec = rearrange(
                rec_feature, "(h w) b c -> b c h w", h=self.feature_size[0]
            )  # B x C X H x W
            
            #! loss 계산 
            pred = torch.sqrt(
                torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True)
            )  # B x 1 x H x W
            pred = self.upsample(pred)  # B x 1 x H x W   
            return {
                "feature_rec": feature_rec,
                "feature_align": feature_align,
                "pred": pred,
                "middle_decoder_feature_0": middle_decoder_feature_rec_0,
                "middle_decoder_feature_1": middle_decoder_feature_rec_1,
                "middle_decoder_feature_2": middle_decoder_feature_rec_2,
            }
            
        else: # Inference
            if task_id is None:
                B = feature_align.shape[0]
                task_id = torch.full((B,), self.null_token_idx, dtype=torch.long, device=device)
            

            # --- Conditional Pass ---
            # task_emb_cond = self.task_embedding(task_id)
            # task_emb_cond_proj = self.task_proj(task_emb_cond)
            # features_cond = src + task_emb_cond_proj.unsqueeze(0) # L, B, C
            # rec_tokens_cond, _ = self.transformer(features_cond, pos_embed) # L, B, C
            # rec_tokens_cond = rec_tokens_cond[3]
            
            #! --- Weight Conditional Pass ---
            task_embedding = self.task_proj(self.task_embedding.weight)
            weight = F.softmax((F.normalize(task_embedding,p=2,dim=1) @ F.normalize(src.mean(0),p=2,dim=1).T),dim=0)
            task_emb_cond_proj = (weight.T @ task_embedding)
            
            #! --- Task 임베딩을 src에 더하기 ---
            # features_cond = src + task_emb_cond_proj.unsqueeze(0) # L, B, C
            features_cond = src
            rec_tokens_cond, _ = self.transformer(features_cond, pos_embed, task_emb_cond_proj.unsqueeze(0)) # L, B, C
            rec_tokens_cond = rec_tokens_cond[3]

            # --- Unconditional Pass ---
            null_task_id = torch.full_like(task_id, self.null_token_idx)
            task_emb_uncond = self.task_embedding(null_task_id)
            task_emb_uncond_proj = self.task_proj(task_emb_uncond)
            
            #! --- Task 임베딩을 src에 더하기 ---
            #features_uncond = src + task_emb_uncond_proj.unsqueeze(0) # L, B, C
            features_uncond = src
            rec_tokens_uncond, _ = self.transformer(features_uncond, pos_embed, task_emb_uncond_proj.unsqueeze(0)) # L, B, C
            rec_tokens_uncond = rec_tokens_uncond[3] 
            
            # --- Apply CFG Formula ---
            guided_tokens_lbc = rec_tokens_uncond + self.guidance_scale * (rec_tokens_cond - rec_tokens_uncond)

            # --- 출력 shape 변경: L, B, C -> B, L, C ---
            rec_feature = self.rec_head(guided_tokens_lbc)
            feature_rec = rearrange(
                rec_feature, "(h w) b c -> b c h w", h=self.feature_size[0]
            )  # B x C X H x W
            
            pred = torch.sqrt(
                torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True)
            )  # B x 1 x H x W
            
            pred = self.upsample(pred)  # B x 1 x H x W
            return {
                "feature_rec": feature_rec,
                "feature_align": feature_align,
                "pred": pred
            }

        
        
    def add_jitter(self, feature_tokens, scale, prob):
        if random.uniform(0, 1) <= prob:
            num_tokens, batch_size, dim_channel = feature_tokens.shape
            feature_norms = (
                feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )  # (H x W) x B x 1
            jitter = torch.randn((num_tokens, batch_size, dim_channel)).cuda()
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens