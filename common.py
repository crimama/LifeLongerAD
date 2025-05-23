import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LOGGER = logging.getLogger(__name__)

class NetworkFeatureAggregator(torch.nn.Module):
    """Aggregates features from different layers of a network."""
    def __init__(self, backbone, layers_to_extract_from, device, train_backbone=False):
        super(NetworkFeatureAggregator, self).__init__()
        """Initialization."""
        self.backbone = backbone
        self.layers_to_extract_from = layers_to_extract_from
        self.device = device
        self.train_backbone = train_backbone
        if not train_backbone:
            self.backbone.eval()
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, images, eval=False):
        """Forward pass."""
        self.backbone.eval() if eval else self.backbone.train()
        features = {}
        with torch.set_grad_enabled(self.train_backbone and not eval):
            # Forward pass through the backbone
            feature = self.backbone.forward_features(images)
            # Extract features from the specified layers
            for layer in self.layers_to_extract_from:
                features[layer] = feature[layer]
        return features

    def feature_dimensions(self, input_shape):
        """Returns dimensions of features for different layers."""
        with torch.no_grad():
            # Create a dummy input
            device = next(self.backbone.parameters()).device
            inp = torch.ones(1, *input_shape).to(device)
            features = self.forward(inp, eval=True)
            feature_dimensions = {
                layer: feature.shape[1] for layer, feature in features.items()
            }
        return feature_dimensions

class Preprocessing(torch.nn.Module):
    """Module for preprocessing features."""
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        """Initialization."""
        self.input_dims = input_dims
        self.output_dim = output_dim
        
        self.preprocessing_modules = torch.nn.ModuleDict()
        for layer, dim in input_dims.items():
            self.preprocessing_modules[layer] = torch.nn.Conv2d(
                dim, output_dim, (1, 1), bias=False
            )

    def forward(self, features):
        """Forward pass."""
        _features = []
        for layer, feature in features.items():
            if layer in self.preprocessing_modules:
                _features.append(self.preprocessing_modules[layer](feature))
        return torch.cat(_features, dim=1)

class Aggregator(torch.nn.Module):
    """Feature aggregator module."""
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        """Initialization."""
        self.target_dim = target_dim
        
    def forward(self, features):
        """Forward pass."""
        # Flatten the features
        features = features.reshape(len(features), -1, self.target_dim)
        return features

class RescaleSegmentor:
    """Rescales segmentation masks to target size."""
    def __init__(self, device, target_size):
        self.device = device
        self.target_size = target_size
        
    def convert_to_segmentation(self, patch_scores, features):
        """Convert patch scores to segmentation masks."""
        with torch.no_grad():
            # Reshape scores and features
            features = features.reshape(len(features), -1, features.shape[-1])
            patch_scores = torch.from_numpy(patch_scores).to(self.device)
            
            # Interpolate to target size
            segmented = F.interpolate(
                patch_scores.unsqueeze(1),
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            )
            return segmented.squeeze(1).cpu().numpy(), features.cpu().numpy() 