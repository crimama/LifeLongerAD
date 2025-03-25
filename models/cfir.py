import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16

# 1) Encoder
class PartialViTEncoder(nn.Module):
    def __init__(
        self,
        pretrained=True,
        extract_layers=[3, 6, 9],
        num_classes=100,
        embed_dim=768,
        mid_block=2
    ):
        """
        mid_block: 중간에 공통 연산을 끝낼 블록 번호 (0 <= mid_block <= 12)
        """
        super().__init__()
        self.model = vit_b_16(pretrained=pretrained)
        self.extract_layers = extract_layers
        self.embed_dim = embed_dim
        self.mid_block = mid_block  # 공통 연산 분기점

        self.cls_embed = nn.Embedding(num_classes + 1, embed_dim)  # +1: unconditional
        self.classifier = nn.Linear(embed_dim, num_classes)

    def _run_encoder_layers(self, x, start_idx, end_idx=None) -> (torch.Tensor, list):
        """
        지정한 인덱스 범위의 encoder 레이어들을 순차적으로 적용.
        extract_layers에 해당하는 중간 피처들을 수집.
        """
        intermediate_features = []
        layers = self.model.encoder.layers
        if end_idx is None:
            end_idx = len(layers)
        for idx in range(start_idx, end_idx):
            x = layers[idx](x)
            if idx in self.extract_layers:
                intermediate_features.append(x)
        return x, intermediate_features

    def add_class_embedding(self, x, labels):
        """
        x: (N, seq_len, embed_dim)
        labels: (N,) 혹은 (2N,) tensor, 각 배치에 대응하는 클래스 인덱스
        cls 토큰에 대응하는 위치에 클래스 임베딩을 더해 반환.
        """
        x_cls = x[:, :1, :]
        class_embed = self.cls_embed(labels).unsqueeze(1)
        return torch.cat([x_cls + class_embed, x[:, 1:, :]], dim=1)

    def partial_forward_common(self, x):
        """
        Encoder의 앞부분(0 ~ mid_block-1)을 공통으로 수행.
        전처리, pos embedding, cls 토큰 추가 후 mid_block 전까지의 레이어 적용.
        """
        x = self.model._process_input(x)
        n = x.shape[0]
        # class token과 pos embedding 적용
        x_cls = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([x_cls, x], dim=1)
        x = self.model.encoder.dropout(x + self.model.encoder.pos_embedding)
        # 0 ~ mid_block-1 레이어 적용 (여기서는 intermediate feature 수집 여부는 필요에 따라 추후 조정)
        x, _ = self._run_encoder_layers(x, start_idx=0, end_idx=self.mid_block)
        return x

    def partial_forward_branch(self, x_both):
        """
        Encoder의 나머지 뒷부분(mid_block ~ end)을 수행.
        extract_layers에 해당하는 피처들을 수집.
        """
        x_both, intermediate_features = self._run_encoder_layers(x_both, start_idx=self.mid_block)
        return x_both, intermediate_features

    def expand_classes(self, new_num_classes):
        # 클래스 확장 로직 (기존과 동일)
        old_weight = self.cls_embed.weight.data
        new_embed = nn.Embedding(new_num_classes + 1, self.embed_dim)
        new_embed.weight.data[: old_weight.size(0)] = old_weight
        self.cls_embed = new_embed

        old_classifier_weight = self.classifier.weight.data
        old_classifier_bias = self.classifier.bias.data
        new_classifier = nn.Linear(self.embed_dim, new_num_classes)
        new_classifier.weight.data[: old_classifier_weight.size(0)] = old_classifier_weight
        new_classifier.bias.data[: old_classifier_bias.size(0)] = old_classifier_bias
        self.classifier = new_classifier

    def forward(self, x, class_labels=None, guidance_prob=0.1):
        """
        단독 forward (기존 로직)
        """
        # 공통 forward
        x = self.partial_forward_common(x)
        n = x.shape[0]
        # cls 토큰 추출
        cls_token = x[:, 0]  # (N, embed_dim)
        # 레이블 결정 (입력이 없으면 classifier를 통해 예측)
        if class_labels is None:
            predicted_classes = self.classifier(cls_token).argmax(dim=1)
            masked_labels = predicted_classes
        else:
            mask = torch.rand(n, device=x.device) < guidance_prob
            masked_labels = class_labels.clone()
            masked_labels[mask] = self.cls_embed.num_embeddings - 1  # unconditional

        # 클래스 임베딩 적용
        x = self.add_class_embedding(x, masked_labels)
        # 나머지 블록 진행 (mid_block부터 끝까지)
        x, _ = self._run_encoder_layers(x, start_idx=self.mid_block)
        return x

# 2) Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=12, embed_dim=768, num_heads=12, mlp_dim=3072, num_patches=196,
                 extract_layers=[3, 6, 9]):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.query_pos = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.reconstruct_head = nn.Linear(embed_dim, 3 * 16 * 16)  # 예시: 16x16 patch 복원
        self.extract_layers = extract_layers

    def forward(self, encoder_features: list):
        """
        encoder_features: list of intermediate features, 각 tensor: (N, seq_len, embed_dim)
        만약 2N batch (cond+uncond)도 그대로 받으면 (2N, seq_len, embed_dim)
        """
        # 예시에서는 단순히 평균으로 fusion 처리 (여러 intermediate feature를 결합)
        fusion_feature = torch.mean(torch.stack(encoder_features), dim=0)
        tgt = self.query_pos.expand(fusion_feature.shape[0], -1, -1)
        intermediate_features = []
        for idx, layer in enumerate(self.decoder.layers):
            tgt = layer(tgt, fusion_feature)
            if idx in self.extract_layers:
                intermediate_features.append(tgt)
        recon_img = self.reconstruct_head(tgt[:, 1:, :])
        # recon_img: (N, patch_num, 3*16*16)
        return recon_img, intermediate_features

# 3) 통합 모델 (CFIR)
class CFIR(nn.Module):
    def __init__(self, backbone='ViT-B-16', input_size=[224, 224], initial_classes=1, 
                 extract_layers=[3, 6, 9], mid_block=2,
                 guidance_scale=3):
        """
        mid_block: Encoder에서 공통 연산을 끊을 위치
        input_size: 원본 이미지 크기 (H, W)
        """
        super().__init__()
        self.input_size = input_size
        self.patch_size = 16  # 고정 patch size

        self.encoder = PartialViTEncoder(
            pretrained=True,
            extract_layers=extract_layers,
            num_classes=initial_classes,
            embed_dim=768,
            mid_block=mid_block
        )
        # # Backbone 부분만 동결하고, classifier 및 cls_embed는 학습하도록 설정
        # for name, param in self.encoder.named_parameters():
        #     if 'cls_embed' in name or 'classifier' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        self.decoder = TransformerDecoder(num_layers=12, num_heads=12, embed_dim=768, extract_layers=extract_layers)
        self.guidance_scale = guidance_scale
        
    def incremental_learning_step(self, new_classes):
        # 클래스 확장
        total_classes = (self.encoder.cls_embed.num_embeddings - 1) + new_classes
        self.encoder.expand_classes(total_classes)

    def split_intermediate(self, N: int, intermediate_features: list):
        inter_cond = []
        inter_uncond = []
        for f in intermediate_features:
            inter_cond.append(f[:N])
            inter_uncond.append(f[N:])
        return inter_cond, inter_uncond

    def patches_to_image(self, patches):
        """
        patches: (N, num_patches, 3*patch_size*patch_size)
        원본 이미지 크기로 재구성 (N, 3, H, W)
        """
        N, num_patches, dim = patches.shape
        H, W = self.input_size
        h_p = H // self.patch_size
        w_p = W // self.patch_size
        # num_patches가 h_p * w_p와 일치하는지 확인
        assert num_patches == h_p * w_p, "Patch 개수가 이미지 크기에 맞지 않습니다."
        patches = patches.view(N, h_p, w_p, 3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        patches = patches.view(N, 3, H, W)
        return patches

    # (A) 기존 single-forward (조건부/무조건부 2회)
    def forward_with_guidance(self, x, class_labels=None):
        # 조건부
        encoder_cond = self.encoder(x, class_labels=class_labels, guidance_prob=0.0)
        # 무조건부
        encoder_uncond = self.encoder(x, class_labels=None, guidance_prob=1.0)
        recon_cond, _ = self.decoder([encoder_cond])
        recon_uncond, _ = self.decoder([encoder_uncond])
        recon_img = recon_uncond + self.guidance_scale * (recon_cond - recon_uncond)
        return recon_img

    # (B) 효율화된 forward (공통 + 분기)
    def forward(self, x, class_labels=None):
        """
        (1) Encoder 전반부 공통 → (2) Cond/Uncond 2배 확장 → (3) Encoder 후반부 →
        (4) Decoder 한 번에 처리 → (5) Guidance 적용.
        intermediate feature들도 함께 반환.
        """
        # 1) Encoder 전반부
        x_mid = self.encoder.partial_forward_common(x)
        n = x_mid.shape[0]
        cls_token = x_mid[:, 0]

        # 2) cond/uncond 라벨 결정
        if class_labels is None:
            predicted_classes = self.encoder.classifier(cls_token).argmax(dim=1)
            class_labels_cond = predicted_classes
        else:
            class_labels_cond = class_labels
        class_labels_uncond = torch.full_like(class_labels_cond, self.encoder.cls_embed.num_embeddings - 1)

        # 3) 2배 확장 (2N) 및 클래스 임베딩 적용
        x_both = torch.cat([x_mid, x_mid], dim=0)
        labels_both = torch.cat([class_labels_cond, class_labels_uncond], dim=0)
        x_cls_both = x_both[:, :1, :]
        batch_class_embed = self.encoder.cls_embed(labels_both).unsqueeze(1)
        x_both = torch.cat([x_cls_both + batch_class_embed, x_both[:, 1:, :]], dim=1)

        # 4) Encoder 후반부 (branch)
        x_final_both, encoder_inter_both = self.encoder.partial_forward_branch(x_both)
        encoder_inter_cond, encoder_inter_uncond = self.split_intermediate(N=n, intermediate_features=encoder_inter_both)

        # 5) Decoder (2N) - intermediate feature 추출
        recon_both, decoder_inter_both = self.decoder(encoder_inter_both)
        recon_cond = recon_both[:n]
        recon_uncond = recon_both[n:]
        decoder_inter_cond, decoder_inter_uncond = self.split_intermediate(N=n, intermediate_features=decoder_inter_both)

        # 6) Guidance 적용
        recon_img = recon_uncond + self.guidance_scale * (recon_cond - recon_uncond)
        return (
            recon_img,
            encoder_inter_cond,
            decoder_inter_cond,
            encoder_inter_uncond,
            decoder_inter_uncond
        )

    def criterion(self, output: list, target_img=None, recon_weight=1.0):
        """
        output: (recon_img, encoder_inter_cond, decoder_inter_cond, encoder_inter_uncond, decoder_inter_uncond)
        target_img: 원본 이미지 (N, 3, H, W). 주어지면 reconstruction loss를 계산.
        """
        loss_sum = 0.0
        cos_loss = nn.CosineSimilarity(dim=1)
        recon_img, enc_inter_cond, dec_inter_cond, enc_inter_uncond, dec_inter_uncond = output

        # encoder intermediate feature에 대한 loss 계산
        for feat_cond, feat_uncond in zip(enc_inter_cond, enc_inter_uncond):
            feat_cond = feat_cond.reshape(feat_cond.shape[0], -1)
            feat_uncond = feat_uncond.reshape(feat_uncond.shape[0], -1)
            loss_encoder = torch.mean(1 - cos_loss(feat_cond, feat_uncond))
            loss_sum += loss_encoder

        # decoder intermediate feature에 대한 loss 계산
        for feat_cond, feat_uncond in zip(dec_inter_cond, dec_inter_uncond):
            feat_cond = feat_cond.reshape(feat_cond.shape[0], -1)
            feat_uncond = feat_uncond.reshape(feat_uncond.shape[0], -1)
            loss_decoder = torch.mean(1 - cos_loss(feat_cond, feat_uncond))
            loss_sum += loss_decoder

        # reconstruction loss 추가 (patch 단위 recon_img를 원본 이미지로 복원하여 계산)
        if target_img is not None:
            # recon_img: (N, num_patches, 3*patch_size*patch_size)
            recon_img_img = self.patches_to_image(recon_img)
            loss_recon = F.l1_loss(recon_img_img, target_img)
            loss_sum += recon_weight * loss_recon

        # guidance penalty (옵션)
        for feat_cond_enc, feat_uncond_enc in zip(enc_inter_cond, enc_inter_uncond):
            diff_enc = torch.mean(torch.abs(feat_cond_enc - feat_uncond_enc))
            loss_sum += self.guidance_scale * diff_enc

        for feat_cond_dec, feat_uncond_dec in zip(dec_inter_cond, dec_inter_uncond):
            diff_dec = torch.mean(torch.abs(feat_cond_dec - feat_uncond_dec))
            loss_sum += self.guidance_scale * diff_dec

        return loss_sum

    
    def get_score_map(self, outputs:list):
        (recon_img,encoder_inter_cond,decoder_inter_cond,encoder_inter_uncond,decoder_inter_uncond) = outputs 

        B,N,D = decoder_inter_cond[0].shape

        Amap = torch.zeros(B,N-1)
        for d_i, e_i in zip(decoder_inter_cond, encoder_inter_cond):
            score = torch.pow(d_i[:,1:,:] - e_i[:,1:,:],2).mean(-1)
            Amap += score.detach().cpu() 
            
        Amap = Amap.view(B,int((N-1)**0.5),int((N-1)**0.5))
        Amap = F.interpolate(Amap.unsqueeze(1), size=tuple(self.input_size), mode='bilinear', align_corners=False).squeeze(1)
        
        return Amap 