import timm
import torch
import torch.nn as nn
from einops import rearrange

class VisionTransformerBackbone(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, intermediate_layers=None):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.intermediate_layers = intermediate_layers or []  # List of block numbers to extract features from

    def get_outplanes(self):
        return [self.model.head.in_features for _ in self.intermediate_layers]
    
    def get_outstrides(self):
        return [self.model.patch_embed.patch_size[0] for _ in self.intermediate_layers]

    def forward(self, Inputs):
        x = Inputs['image']
        B = x.shape[0]
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        
        intermediates = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in self.intermediate_layers:                
                # Directly rearrange and append without storing intermediate x_without_cls
                h = int((x.shape[1] - 1) ** 0.5)  # Calculate h,w from sequence length (excluding CLS token)
                intermediates.append(
                    rearrange(x[:, 1:, :], 'b (h w) c -> b c h w', h=h, w=h)
                )
                
        x = self.model.norm(x)
        
        output = {            
            'features': [feat[:, :, :] for feat in intermediates]  # Exclude CLS token (first token) from each feature
        }
        
        return output