import torch
import torch.nn.functional as F
import einops
from timm import create_model

class VitBackbone(torch.nn.Module):
    def __init__(self, model_name='vit_base_patch16_224.orig_in21k', device='cuda'):
        super(VitBackbone, self).__init__()
        self.is_vit = True
        self.model = create_model(model_name, pretrained=True).to(device)
        
        # Set all parameters to non-trainable
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x, layers_to_extract=None):
        if layers_to_extract is None:
            # Default: extract features from all transformer blocks
            layers_to_extract = list(range(len(self.model.blocks)))
        
        # Initial processing
        x = self.model.patch_embed(x)
        
        # Add positional embedding without cls token
        if hasattr(self.model, 'pos_embed'):
            # Skip the class token position (index 0)
            pos_embed = self.model.pos_embed[:, 1:, :]
            x = x + pos_embed
        
        x = self.model.pos_drop(x)
        
        # Store intermediate features
        features = []        
        # Process through transformer blocks
        for i, block in enumerate(self.model.blocks):
            x = block(x)
            if i in layers_to_extract:
                features.append(x)
        
        # Process features to correct shape
        for j in range(len(features)):
            B, PxP, C = features[j].shape
            
            # Calculate P (assuming square patches)
            P = int(PxP ** 0.5)
            
            # Reshape to spatial format
            features[j] = einops.rearrange(features[j], 'b (h w) c -> b c h w', h=P, w=P)
            
        return features