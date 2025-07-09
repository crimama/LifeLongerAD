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

class CL_PromptInput():

    def __init__(self, model, device, 
                 use_pgpt=False, prompt_dim=256):
        """
        Initializes the Continual Learning manager with class-level prompt support.

        Args:
            model (nn.Module): The neural network model.
            device (torch.device): The device to run computations on (CPU or GPU).
            use_pgpt (bool): Whether to use PGPT (Prototype-Guided Prompt Tuning). Defaults to False.
            prompt_dim (int): Dimension of prompt vectors. Defaults to 256.
        """
        self.model = model
        self.device = device
        self.current_task = 0
        self.use_pgpt = use_pgpt
        self.prompt_dim = prompt_dim
        if self.use_pgpt:
            self.prompt_pool = {}  # {class_name: [prompt_layer1, prompt_layer2, ...]}
            self.prototype_repository = {}  # {class_name: prototype_tensor}
            self.current_class_name = None
            self.active_classes = set()  # Track classes in current task for batch processing
            print(f"PGPT enabled with class-level prompts: prompt_dim={prompt_dim}")

    # --- PGPT (Prototype-Guided Prompt Tuning) Methods ---
    
    def initialize_prompt_for_class(self, class_name):
        """Initialize prompts for a new class."""
        if not self.use_pgpt:
            return
            
        if class_name not in self.prompt_pool:
            # Initialize random prompts for the new class            
            num_queries = self.model.reconstruction.feature_size[0]**2
            num_layers = self.model.reconstruction.transformer.decoder.num_layers
            prompts = [nn.Embedding(num_queries, self.prompt_dim).to(self.device) for _ in range(num_layers)]            
            
            self.prompt_pool[class_name] = prompts
            self.active_classes.add(class_name)
            print(f"Initialized {num_layers} prompt layers with {num_queries} queries each for class '{class_name}'")
    
    def set_current_class(self, class_name):
        """Set the current primary class for training."""
        if not self.use_pgpt:
            return
            
        self.current_class_name = class_name
        # Don't inject prompts here anymore - will be done dynamically per batch
        print(f"Set current primary class: {class_name}")
    
    def add_class_to_task(self, class_name):
        """Add a class to the current task's active classes."""
        if not self.use_pgpt:
            return
            
        self.initialize_prompt_for_class(class_name)
        self.active_classes.add(class_name)
        print(f"Added class '{class_name}' to current task. Active classes: {self.active_classes}")
    
    def get_prompts_for_class(self, class_name):
        """Get prompts for a specific class."""
        if not self.use_pgpt or class_name not in self.prompt_pool:
            return None
        return self.prompt_pool[class_name]
    
    def get_class_from_label(self, class_label):
        """Convert class label to class name. Override this based on your label mapping."""
        # This is a placeholder - you should implement based on your class_label_mapping
        from datasets.mvtecad import class_label_mapping
        
        # Reverse lookup from label to class name
        for class_name, label in class_label_mapping.items():
            if label == class_label:
                return class_name
        return None
    
    def inject_prompts_for_batch(self, class_labels, backbone_features=None):
        """
        Inject appropriate prompts for each sample in the batch based on their class labels.
        Enhanced to handle cross-class evaluation with K-NN fallback.
        
        Args:
            class_labels: Tensor of shape [B] containing class labels for each sample in batch
            backbone_features: Optional tensor of backbone features for K-NN selection fallback
        """
        if not self.use_pgpt:
            return
            
        batch_size = class_labels.shape[0]
        
        # Get unique classes in this batch
        unique_labels = torch.unique(class_labels)
        
        # For each layer, we need to handle prompts for each unique class
        num_layers = self.model.reconstruction.transformer.decoder.num_layers
        
        for layer_idx in range(num_layers):
            # Create a batch-aware prompt tensor
            batch_prompts = []
            
            for batch_idx in range(batch_size):
                sample_label = class_labels[batch_idx].item()
                class_name = self.get_class_from_label(sample_label)
                
                if class_name and class_name in self.prompt_pool:
                    # Use class-specific prompt
                    class_prompts = self.prompt_pool[class_name]
                    sample_prompt = class_prompts[layer_idx].weight  # [num_queries, prompt_dim]
                    batch_prompts.append(sample_prompt)
                else:
                    # Enhanced fallback logic with K-NN selection
                    fallback_prompt = None
                    
                    # Try K-NN selection if features are available
                    if backbone_features is not None and batch_idx < backbone_features.shape[0] and self.prototype_repository:
                        try:
                            sample_features = backbone_features[batch_idx].mean(0).reshape(backbone_features.shape[1], -1)
                            knn_prompts, selected_class = self.select_prompt_by_knn(sample_features)
                            if knn_prompts and layer_idx < len(knn_prompts):
                                fallback_prompt = knn_prompts[layer_idx].weight
                                # Log K-NN selection for debugging
                                if batch_idx == 0:  # Only log for first sample to avoid spam
                                    print(f"K-NN fallback: selected '{selected_class}' for class '{class_name}' (layer {layer_idx})")
                        except Exception as e:
                            print(f"K-NN selection failed for sample {batch_idx}: {e}")
                    
                    # If K-NN didn't work, use current class prompt
                    if fallback_prompt is None and self.current_class_name and self.current_class_name in self.prompt_pool:
                        fallback_prompts = self.prompt_pool[self.current_class_name]
                        fallback_prompt = fallback_prompts[layer_idx].weight
                    
                    # Last resort: create zero prompt
                    if fallback_prompt is None:
                        num_queries = self.model.reconstruction.feature_size[0]**2
                        fallback_prompt = torch.zeros(num_queries, self.prompt_dim, device=self.device)
                        if batch_idx == 0:  # Only log once per batch
                            print(f"Warning: Using zero prompt for class '{class_name}' (no learned prompt available)")
                    
                    batch_prompts.append(fallback_prompt)
            
            # Stack prompts for all samples in batch: [B, num_queries, prompt_dim]
            batch_prompt_tensor = torch.stack(batch_prompts, dim=0)
            
            # Create a custom embedding layer for this batch
            batch_prompt_embedding = BatchPromptEmbedding(batch_prompt_tensor)
            
            # Inject into model
            self.model.reconstruction.transformer.decoder.layers[layer_idx].learned_embed = batch_prompt_embedding
    
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
                Input = {'image':images,'clslabel':class_labels}
                
                # Filter samples that belong to the target class
                target_class_label = None
                from datasets.mvtecad import class_label_mapping
                if class_name in class_label_mapping:
                    target_class_label = class_label_mapping[class_name]
                
                if target_class_label is not None:
                    # Only use samples from the target class
                    mask = (class_labels == target_class_label)
                    if mask.sum() > 0:
                        filtered_images = images[mask]
                        filtered_class_labels = class_labels[mask]
                        filtered_input = {'image': filtered_images, 'clslabel': filtered_class_labels}
                        
                        # Extract features from backbone
                        backbone_features = self.model.backbone(filtered_input)                
                        backbone_features = self.model.neck(backbone_features)
                        backbone_features = backbone_features.get('feature_align', None)                                                
                        
                        if backbone_features is not None:
                            features.append(backbone_features.cpu())
                
                # Limit the number of batches for prototype calculation
                if batch_idx >= 5:  # Use first 5 batches for prototype
                    break
        
        if features:
            # Calculate mean prototype
            all_features = torch.cat(features, dim=0)
            prototype = all_features.mean(0).reshape(all_features.shape[1],-1)
            self.prototype_repository[class_name] = prototype.cpu()
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
            # Return all prompts for the class (list of embeddings for each layer)
            return prompts, best_class
        
        return None, None
    
    def inject_prompt_into_model(self, prompt):
        """Inject prompt into the model input for reconstruction (legacy method for single-class batches)."""
        for i, p in enumerate(prompt):    
            # 기존 learned_embed 파라미터를 새로운 embedding으로 교체
            p = p.to(self.device)
            self.model.reconstruction.transformer.decoder.layers[i].learned_embed = p
            
            # 새로운 파라미터를 모델의 파라미터 그래프에 등록
            # 이렇게 해야 gradient가 계산되고 optimizer가 업데이트할 수 있음
            self.model.reconstruction.transformer.decoder.layers[i].add_module(f'learned_embed', p)                         

    # --- Basic Continual Learning Methods ---

    def prepare_next_task(self):
        """Prepares the network state for the next task."""
        print(f"\n--- Preparing for Task {self.current_task + 1} ---")
        
        # Clear active classes for new task
        self.active_classes.clear()
        self.current_class_name = None
        
        self.current_task += 1
        print(f"--- Ready for Task {self.current_task} ---")
        return True
    

    def state_dict(self):
        """Save the state of the continual learning manager."""
        state = {
            'current_task': self.current_task,
            'use_pgpt': self.use_pgpt,
            'prompt_dim': self.prompt_dim,
            'current_class_name': self.current_class_name,
            'active_classes': list(self.active_classes),
            'prototype_repository': self.prototype_repository
        }
        
        # Save prompt_pool state_dicts properly
        if self.use_pgpt and self.prompt_pool:
            prompt_states = {}
            for class_name, prompts in self.prompt_pool.items():
                prompt_states[class_name] = [prompt.state_dict() for prompt in prompts]
            state['prompt_pool_states'] = prompt_states
            
        return state
    
    def load_state_dict(self, state_dict):
        """Load the state of the continual learning manager."""
        self.current_task = state_dict.get('current_task', 0)
        self.use_pgpt = state_dict.get('use_pgpt', False)
        self.prompt_dim = state_dict.get('prompt_dim', 256)
        self.current_class_name = state_dict.get('current_class_name', None)
        self.active_classes = set(state_dict.get('active_classes', []))
        self.prototype_repository = state_dict.get('prototype_repository', {})
        
        # Restore prompt_pool from saved states
        if self.use_pgpt and 'prompt_pool_states' in state_dict:
            self.prompt_pool = {}
            for class_name, prompt_states in state_dict['prompt_pool_states'].items():
                # Reconstruct prompts from saved states
                prompts = []
                for prompt_state in prompt_states:
                    # Get dimensions from saved state
                    weight_shape = prompt_state['weight'].shape
                    num_embeddings, embedding_dim = weight_shape
                    
                    # Create new embedding and load state
                    prompt = nn.Embedding(num_embeddings, embedding_dim).to(self.device)
                    prompt.load_state_dict(prompt_state)
                    prompts.append(prompt)
                
                self.prompt_pool[class_name] = prompts
                print(f"Restored {len(prompts)} prompts for class '{class_name}'")
        
        print(f"CL Manager state loaded: task={self.current_task}, classes={len(self.prompt_pool) if self.use_pgpt else 0}")


class BatchPromptEmbedding(nn.Module):
    """
    Custom embedding layer that handles batch-wise prompts.
    Each sample in the batch can have different prompt weights.
    """
    def __init__(self, batch_prompts):
        """
        Args:
            batch_prompts: Tensor of shape [B, num_queries, prompt_dim]
        """
        super().__init__()
        self.batch_prompts = batch_prompts.clone()  # [B, num_queries, prompt_dim]
        
    def forward(self, input_ids):
        """
        Args:
            input_ids: Tensor of shape [B, seq_len] containing token indices
        Returns:
            Tensor of shape [B, seq_len, prompt_dim]
        """
        batch_size, seq_len = input_ids.shape
        
        # Gather embeddings for each sample in batch
        outputs = []
        for b in range(batch_size):
            # Get prompt weights for this sample: [num_queries, prompt_dim]
            sample_prompts = self.batch_prompts[b]
            
            # Use F.embedding to lookup embeddings
            sample_output = F.embedding(input_ids[b], sample_prompts)  # [seq_len, prompt_dim]
            outputs.append(sample_output)
        
        # Stack outputs: [B, seq_len, prompt_dim]
        return torch.stack(outputs, dim=0)
    
    @property
    def weight(self):
        """Compatibility property for standard embedding access."""
        # Return the first sample's prompts as default
        return self.batch_prompts[0] if len(self.batch_prompts) > 0 else None        
