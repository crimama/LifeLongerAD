# Author: Ghada Sokar et al. (Original Author), Modified for Transformer & Reconstruction Adaptation
# This is a modified implementation based on the SpaceNet paper for Continual Learning,
# adapted for Transformer-like architectures and reconstruction tasks by removing
# classifier-specific logic and treating the final layer like internal layers.
# Grow strategy is modified to use weight importance as a proxy for gradient information.
# Original Paper Citation:
# @article{SOKAR20211,
# title = {SpaceNet: Make Free Space for Continual Learning},
# journal = {Neurocomputing},
# volume = {439},
# pages = {1-11},
# year = {2021},
# issn = {0925-2312},
# doi = {https://doi.org/10.1016/j.neucom.2021.01.078},
# url = {https://www.sciencedirect.com/science/article/pii/S0925231221001545},
# author = {Ghada Sokar and Decebal Constantin Mocanu and Mykola Pechenizkiy}
# }

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import math
from collections.abc import Mapping # Import Mapping for type checking in to_device if needed

class CL_Transformer():

    def __init__(self, model, device, 
                 use_pgpt=False, prompt_dim=256, num_prompts_per_class=1, k_neighbors=1):
        """
        Initializes the Continual Learning manager.

        Args:
            model (nn.Module): The neural network model.
            device (torch.device): The device to run computations on (CPU or GPU).
            use_pgpt (bool): Whether to use PGPT (Prototype-Guided Prompt Tuning). Defaults to False.
            prompt_dim (int): Dimension of prompt vectors. Defaults to 256.
            num_prompts_per_class (int): Number of prompts per class. Defaults to 1.
            k_neighbors (int): Number of neighbors for K-NN selection. Defaults to 1.
        """
        self.model = model
        self.device = device
        self.current_task = 0
        self.use_pgpt = use_pgpt
        self.prompt_dim = prompt_dim
        self.num_prompts_per_class = num_prompts_per_class
        self.k_neighbors = k_neighbors
        if self.use_pgpt:
            self.prompt_pool = {}
            self.prototype_repository = {}
            self.current_class_name = None
            self.feature_extractor = None
            print(f"PGPT enabled: prompt_dim={prompt_dim}, prompts_per_class={num_prompts_per_class}, k_neighbors={k_neighbors}")
        
        # Store initial weights for basic continual learning
        self.init_weights = {}
        self.old_weights = {}

    # --- PGPT (Prototype-Guided Prompt Tuning) Methods ---
    
    def initialize_prompt_for_class(self, class_name):
        """Initialize prompts for a new class."""
        if not self.use_pgpt:
            return
            
        if class_name not in self.prompt_pool:
            # Initialize random prompts for the new class
            prompts = []
            for i in range(self.num_prompts_per_class):
                prompt = torch.randn(self.prompt_dim, device=self.device) * 0.02  # Small initialization
                prompts.append(prompt)
            
            self.prompt_pool[class_name] = prompts
            print(f"Initialized {self.num_prompts_per_class} prompts for class '{class_name}'")
    
    def set_current_class(self, class_name):
        """Set the current class for training and freeze other prompts."""
        if not self.use_pgpt:
            return
            
        self.current_class_name = class_name
        
        # Freeze all prompts except the current class
        for name, param in self.model.named_parameters():
            if 'prompt' in name.lower():
                if class_name in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        print(f"Set current class: '{class_name}', other prompts frozen")
    
    def get_prompts_for_class(self, class_name):
        """Get prompts for a specific class."""
        if not self.use_pgpt or class_name not in self.prompt_pool:
            return None
        return self.prompt_pool[class_name]
    
    def calculate_prototype(self, dataloader, class_name):
        """Calculate prototype features for a class using the current backbone."""
        if not self.use_pgpt:
            return
            
        print(f"Calculating prototype for class '{class_name}'...")
        
        self.model.eval()
        features = []
        
        with torch.no_grad():
            for batch_idx, (images, labels, class_labels) in enumerate(dataloader):
                # Move data to device
                images = images.to(self.device)
                
                # Extract features from backbone (assuming the model has a backbone attribute)
                # This depends on the specific model architecture
                if hasattr(self.model, 'backbone'):
                    backbone_features = self.model.backbone(images)
                else:
                    # Fallback: use the first part of the model for feature extraction
                    # This needs to be adapted based on the actual model structure
                    backbone_features = self.model(images)
                    if isinstance(backbone_features, dict):
                        backbone_features = backbone_features.get('feature_align', backbone_features)
                
                # Average pooling if needed
                if backbone_features.dim() > 2:
                    backbone_features = F.adaptive_avg_pool2d(backbone_features, (1, 1)).squeeze(-1).squeeze(-1)
                
                features.append(backbone_features.cpu())
                
                # Limit the number of batches for prototype calculation
                if batch_idx >= 10:  # Use first 10 batches for prototype
                    break
        
        if features:
            # Calculate mean prototype
            prototype = torch.cat(features, dim=0).mean(dim=0)
            self.prototype_repository[class_name] = prototype.to(self.device)
            print(f"Prototype calculated for '{class_name}': {prototype.shape}")
        else:
            print(f"Warning: No features extracted for prototype calculation of '{class_name}'")
    
    def select_prompt_by_knn(self, input_features):
        """Select the best prompt using K-NN on prototypes."""
        if not self.use_pgpt or not self.prototype_repository:
            return None, None
            
        # Calculate distances to all prototypes
        distances = {}
        input_features = input_features.to(self.device)
        
        for class_name, prototype in self.prototype_repository.items():
            prototype = prototype.to(self.device)
            distance = F.cosine_similarity(input_features.unsqueeze(0), prototype.unsqueeze(0), dim=1)
            distances[class_name] = distance.item()
        
        # Find the closest class
        best_class = min(distances, key=distances.get)
        
        # Get prompts for the best class
        prompts = self.get_prompts_for_class(best_class)
        if prompts:
            # For simplicity, return the first prompt
            # In a more sophisticated version, you could ensemble multiple prompts
            return prompts[0], best_class
        
        return None, None
    
    def inject_prompt_into_model(self, prompt, model_input):
        """Inject prompt into the model input for reconstruction."""
        if not self.use_pgpt or prompt is None:
            return model_input
            
        # This method needs to be adapted based on the specific model architecture
        # For now, we'll assume the model can handle prompt injection
        if isinstance(model_input, dict):
            model_input['prompt'] = prompt
        else:
            # If model_input is a tensor, we need to modify the model to handle prompts
            print("Warning: Prompt injection not implemented for tensor input")
            
        return model_input

    # --- Basic Continual Learning Methods ---
    
    def set_init_network_weight(self):
        """Stores the initial weights for basic continual learning."""
        self.init_weights = {}
        with torch.no_grad():
            print("Storing initial weights...")
            for name, param in self.model.named_parameters():
                self.init_weights[name] = copy.deepcopy(param.data)

    def save_old_tasks_weights(self):
        """Saves the current weights before an optimizer step."""
        self.old_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.old_weights[name] = copy.deepcopy(param.data)

    def recover_old_tasks_weights(self):
        """Basic weight recovery - placeholder for compatibility."""
        pass

    def apply_mask_on_grad(self):
        """Placeholder for gradient masking - removed DST functionality."""
        pass

    def reset_importance(self):
        """Placeholder for importance reset - removed DST functionality."""
        pass

    def prepare_next_task(self):
        """Prepares the network state for the next task."""
        print(f"\n--- Preparing for Task {self.current_task + 1} ---")        
        self.current_task += 1
        print(f"--- Ready for Task {self.current_task} ---")
        return True
    
    def save_current_mask(self):
        """Placeholder for mask saving - removed DST functionality."""
        self.current_task += 1

    def set_evaluation_mask(self):
        """Placeholder for evaluation mask - removed DST functionality."""
        print("Evaluation mode - DST masking removed")
        self.model.eval()

