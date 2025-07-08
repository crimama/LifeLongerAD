import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CL_PromptInput():

    def __init__(self, model, device, 
                 use_pgpt=False, prompt_dim=256):
        """
        Initializes the Continual Learning manager.

        Args:
            model (nn.Module): The neural network model.
            device (torch.device): The device to run computations on (CPU or GPU).
            use_pgpt (bool): Whether to use PGPT (Prototype-Guided Prompt Tuning). Defaults to False.
            prompt_dim (int): Dimension of prompt vectors. Defaults to 256.
            num_prompts_per_class (int): Number of prompts per class. Defaults to 1.
        """
        self.model = model
        self.device = device
        self.current_task = 0
        self.use_pgpt = use_pgpt
        self.prompt_dim = prompt_dim
        if self.use_pgpt:
            self.prompt_pool = {}
            self.prototype_repository = {}
            self.current_class_name = None
            print(f"PGPT enabled: prompt_dim={prompt_dim}")

    # --- PGPT (Prototype-Guided Prompt Tuning) Methods ---
    
    def initialize_prompt_for_class(self, class_name):
        """Initialize prompts for a new class."""
        if not self.use_pgpt:
            return
            
        if class_name not in self.prompt_pool:
            # Initialize random prompts for the new class            
            num_queries = self.model.reconstruction.feature_size[0]**2
            num_layers = self.model.reconstruction.transformer.decoder.num_layers
            prompts = [nn.Embedding(num_queries, self.prompt_dim) for _ in range(num_layers)]            
            
            self.prompt_pool[class_name] = prompts
            print(f"Initialized {num_queries} prompts for class '{class_name}'")
    
    def set_current_class(self, class_name):
        """Set the current class for training and freeze other prompts."""
        if not self.use_pgpt:
            return
            
        self.current_class_name = class_name
        current_prompts = self.prompt_pool[class_name]
        self.inject_prompt_into_model(current_prompts)              
        
    
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
                Input = {'image':images,'clslabel':class_labels}
                # Extract features from backbone (assuming the model has a backbone attribute)

                backbone_features = self.model.backbone(Input)                
                backbone_features = self.model.neck(backbone_features)
                backbone_features = backbone_features.get('feature_align', None)                                                
                
                features.append(backbone_features.cpu())
                
                # Limit the number of batches for prototype calculation
                if batch_idx >= 5:  # Use first 10 batches for prototype
                    break
        
        if features:
            # Calculate mean prototype
            prototype = torch.cat(features).mean(0).reshape(backbone_features.shape[1],-1)
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
        """Inject prompt into the model input for reconstruction."""
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
        