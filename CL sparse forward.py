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
    """
    Continual Learning class adapted for Transformer-like architectures and reconstruction tasks.
    Applies dynamic sparse training principles inspired by SpaceNet, focusing
    on connection sparsity within Linear/Conv2d layers. Treats the final layer
    the same as internal layers regarding sparsity management.
    Grow strategy uses accumulated weight importance to select new connections.
    """
    def __init__(self, model, device, sparsity_config, replace_percentage=0.2):
        """
        Initializes the Continual Learning manager.

        Args:
            model (nn.Module): The neural network model.
            device (torch.device): The device to run computations on (CPU or GPU).
            sparsity_config (dict): Configuration for sparsity.
                Example: {'default_sparsity': 0.8, 'layer_specific': {'layers.0.linear1': 0.7, 'reconstruction.rec_head': 0.5}}
                - default_sparsity: Default connection sparsity for layers not specified otherwise.
                - layer_specific: Sparsity override for specific layer names (use full names).
            replace_percentage (float, optional): Percentage of connections to drop and grow in each epoch. Defaults to 0.2.
        """
        self.model = model
        self.device = device
        self.sparsity_config = sparsity_config
        self.replace_percentage = replace_percentage
        self.inf = float('inf')
        self.current_task = 0

        # --- Model Analysis ---
        self.learnable_layers = self._identify_learnable_layers()
        self.param_info = self._get_param_info()

        # --- State Initialization ---
        self.mask = {}  # Current task mask (active connections)
        self.previous_mask = {}  # Previous task mask (accumulated)
        self.init_weights = {}  # Initial weights storage
        self.old_weights = {}  # Previous step weights storage
        self.weights_importance = {}  # Weight importance scores
        self.removed_mask = {}  # Mask for dropped connections
        self.replace_count = {}  # Number of connections to replace

        # Initialize state variables
        self._initialize_state()
        # Create initial masks
        self.create_masks()
        # Apply masks to model parameters
        self._apply_masks_to_model()

    def _identify_learnable_layers(self):
        """Identifies learnable layers (e.g., Linear, Conv2d), ignoring LayerNorm."""
        learnable_layers = []
        print("Identifying learnable layers (focusing on nn.Linear/Conv2d, ignoring nn.LayerNorm)...")
        for name, module in self.model.named_modules():
            if isinstance(module, nn.LayerNorm):
                continue
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                has_params = any(p.requires_grad for p in module.parameters(recurse=False))
                is_valid_layer = True
                if isinstance(module, nn.Linear) and (module.out_features == 0 or module.in_features == 0):
                    is_valid_layer = False
                elif isinstance(module, nn.Conv2d) and (module.out_channels == 0 or module.in_channels == 0):
                     is_valid_layer = False

                if has_params and is_valid_layer:
                     learnable_layers.append({'name': name, 'module': module})

        if not learnable_layers:
             raise ValueError("No learnable layers (Linear, Conv2d) with parameters found in the model.")
        print(f"Identified Learnable Layers: {[layer['name'] for layer in learnable_layers]}")
        return learnable_layers

    def _get_param_info(self):
        """Extracts information about learnable parameters (weights)."""
        param_info = []
        for layer_info in self.learnable_layers:
            layer_name = layer_info['name']
            module = layer_info['module']
            weight_param = None
            weight_param_name = ""
            for param_name, param in module.named_parameters(recurse=False):
                if 'weight' in param_name and param.requires_grad:
                    weight_param = param
                    weight_param_name = f"{layer_name}.{param_name}"
                    break

            if weight_param is not None:
                 param_info.append({
                    'param_name': weight_param_name,
                    'param': weight_param,
                    'layer_name': layer_name,
                    'module': module
                 })
        return param_info

    # Removed _find_classifier and _get_num_classes

    def _initialize_state(self):
        """Initializes masks, importance scores, and other state variables."""
        # Masks and importance scores are created on self.device
        for info in self.param_info:
            param_name = info['param_name']
            param = info['param']
            # Initialize masks on the initially specified device
            self.mask[param_name] = torch.zeros_like(param.data, device=self.device)
            self.previous_mask[param_name] = torch.zeros_like(param.data, device=self.device)
            # Initialize importance scores on the initially specified device
            self.weights_importance[param_name] = torch.zeros_like(param.data, device=self.device)        

    def _get_connection_count(self, param_name):
        """Calculates the number of connections to keep for a parameter based on sparsity config."""
        param = dict(self.model.named_parameters()).get(param_name)
        if param is None:
            print(f"Warning: Parameter '{param_name}' not found in model for connection count.")
            return 0
        layer_name = next((info['layer_name'] for info in self.param_info if info['param_name'] == param_name), None)
        if layer_name is None: layer_name = param_name.split('.')[0] # Fallback

        sparsity = self.sparsity_config.get('layer_specific', {}).get(layer_name, self.sparsity_config.get('default_sparsity', 0.5))
        total_connections = param.numel()
        return int(total_connections * (1.0 - sparsity))

    # --- Masking and Sparsity ---
    def _apply_masks_to_model(self):
        """Apply masks to model parameters and fix unselected parameters."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.mask:
                    # Apply mask to parameter data
                    param.data *= self.mask[name].to(param.device)
                    # Fix unselected parameters
                    param.requires_grad = False
                    # Only enable gradients for selected parameters
                    param.requires_grad = True
                    # Create a hook to apply mask during forward pass
                    param.register_hook(lambda grad, name=name: grad * self.mask[name].to(grad.device))

    def create_masks(self):
        """Creates sparsity masks for the current task."""
        print("Creating masks for current task...")
        for info in self.param_info:
            param_name = info['param_name']
            param = info['param']

            # Reset current task mask
            if param_name not in self.mask:
                self.mask[param_name] = torch.zeros_like(param.data, device=self.device)
            else:
                self.mask[param_name] = self.mask[param_name].to(self.device)
            self.mask[param_name].zero_()

            # Ensure previous mask is on the correct device
            if param_name not in self.previous_mask:
                self.previous_mask[param_name] = torch.zeros_like(param.data, device=self.device)
            else:
                self.previous_mask[param_name] = self.previous_mask[param_name].to(self.device)

            # Uniform Masking for All Layers
            num_to_select = self._get_connection_count(param_name)
            available_connections_mask = (self.previous_mask[param_name] == 0)
            available_indices = torch.where(available_connections_mask.flatten())[0]
            num_available = len(available_indices)
            num_to_select = min(num_to_select, num_available)

            if num_available > 0 and num_to_select > 0:
                perm = torch.randperm(num_available, device=self.device)
                selected_flat_indices = available_indices[perm[:num_to_select]]
                self.mask[param_name].view(-1)[selected_flat_indices] = 1.0
            elif num_to_select > 0:
                print(f"  Param '{param_name}': Warning! Wanted {num_to_select} connections, but 0 were available in non-previous spots.")

        # Apply masks to model parameters
        self._apply_masks_to_model()

    def set_init_network_weight(self):
        """Stores the initial weights and applies the initial mask."""
        self.init_weights = {}
        with torch.no_grad():
            print("Storing initial weights and applying initial masks...")
            for name, param in self.model.named_parameters():
                if name in self.mask:
                    self.init_weights[name] = copy.deepcopy(param.data)
                    current_mask = self.mask[name].to(param.device)
                    param.data *= current_mask
                    self.mask[name] = current_mask
                    # Fix unselected parameters
                    param.requires_grad = False
                    # Only enable gradients for selected parameters
                    param.requires_grad = True
                elif 'bias' in name:
                    self.init_weights[name] = copy.deepcopy(param.data)

    def save_old_tasks_weights(self):
        """Saves the current weights before an optimizer step."""
        self.old_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.old_weights[name] = copy.deepcopy(param.data)

    def recover_old_tasks_weights(self):
        """Recovers weights for connections belonging *only* to previous tasks."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param_device = param.device
                if name in self.previous_mask and name in self.mask and name in self.old_weights:
                    prev_mask = self.previous_mask[name].to(param_device)
                    curr_mask = self.mask[name].to(param_device)
                    recover_mask = (prev_mask == 1) & (curr_mask == 0)
                    if recover_mask.any():
                        old_w = self.old_weights[name].to(param_device)
                        param.data[recover_mask] = old_w[recover_mask]

    def apply_mask_on_grad(self):
        """Applies the current task's mask to gradients."""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_device = param.device
                if name in self.mask:
                    current_mask = self.mask[name].to(param_device)
                    param.grad *= current_mask
                    
                if name in self.previous_mask:
                    protect = self.previous_mask[name].to(param.device)
                    param.grad *= (1.0 - protect)                     

    # --- Importance Calculation ---
    def reset_importance(self):
        """Resets importance scores before training a new task."""
        self.weights_importance = {}
        for info in self.param_info:
             param_name = info['param_name']
             param = info['param']
             self.weights_importance[param_name] = torch.zeros_like(param.data, device=param.device)

    def calculate_importance(self):
        """Calculates weight importance based on gradient and weight change."""
        for name, param in self.model.named_parameters():
            if name in self.weights_importance and name in self.old_weights and param.grad is not None and name in self.mask:
                param_device = param.device
                old_weight = self.old_weights[name].to(param_device)
                grad = param.grad.to(param_device)
                current_mask = self.mask[name].to(param_device)
                self.weights_importance[name] += abs(
                    (param.data - old_weight) * grad * current_mask
                )

    # --- Drop and Grow ---
    def drop(self):
        """Drops connections with the lowest importance for the current task."""
        self.removed_mask = {}
        self.replace_count = {}
        # print("Dropping connections...")
        for info in self.param_info:
            param_name = info['param_name']
            if param_name not in self.mask or param_name not in self.weights_importance: continue

            current_mask = self.mask[param_name].to(self.device)
            num_active = current_mask.sum().item()
            if num_active == 0:
                self.replace_count[param_name] = 0
                continue

            replace_count = int(num_active * self.replace_percentage)
            self.replace_count[param_name] = replace_count
            if replace_count == 0: continue

            importance = copy.deepcopy(self.weights_importance[param_name]).to(self.device)
            importance[current_mask == 0] = self.inf

            flat_importance = importance.flatten()

            try:
                num_valid_to_drop = (flat_importance != self.inf).sum().item()
                k = min(replace_count, max(0, num_valid_to_drop))

                indices_to_remove_absolute = None
                if k > 0:
                    _, indices_to_remove_absolute = torch.topk(flat_importance, k, largest=False, sorted=False)
                    if not torch.isfinite(flat_importance[indices_to_remove_absolute]).all():
                         print(f"Warning: topk in drop for {param_name} returned indices pointing to non-finite values. Retrying with filter.")
                         finite_mask_flat = torch.isfinite(flat_importance)
                         finite_indices = torch.where(finite_mask_flat)[0]
                         if len(finite_indices) >= k:
                              finite_importances = flat_importance[finite_indices]
                              _, topk_indices_relative = torch.topk(finite_importances, k, largest=False)
                              indices_to_remove_absolute = finite_indices[topk_indices_relative]
                         else:
                              print(f"Warning: Not enough finite values ({len(finite_indices)}) to drop {k} for {param_name}.")
                              indices_to_remove_absolute = None
                              self.replace_count[param_name] = 0

                if indices_to_remove_absolute is not None and len(indices_to_remove_absolute) > 0:
                    self.mask[param_name] = self.mask[param_name].to(self.device)
                    self.mask[param_name].view(-1)[indices_to_remove_absolute] = 0.0
                    self.removed_mask[param_name] = indices_to_remove_absolute.cpu()
                else:
                     self.replace_count[param_name] = 0

            except RuntimeError as e:
                 print(f"Error during topk for drop {param_name}: {e}")
                 self.removed_mask[param_name] = None
                 self.replace_count[param_name] = 0

    def grow(self):
        """Grows new connections randomly into available spots."""
        with torch.no_grad():
            # print("Growing connections...")
            for info in self.param_info:
                param_name = info['param_name']

                if param_name not in self.replace_count or \
                   self.replace_count[param_name] == 0 or \
                   param_name not in self.mask or \
                   param_name not in self.previous_mask:
                    continue

                num_to_add = self.replace_count[param_name]
                if num_to_add <= 0: continue

                # --- Find Available Locations ---
                current_mask = self.mask[param_name].to(self.device)
                prev_mask = self.previous_mask[param_name].to(self.device)
                
                # Available = not current_mask AND not previous_mask
                available_mask = (current_mask == 0) & (prev_mask == 0)
                available_indices = torch.where(available_mask.flatten())[0]
                num_available = len(available_indices)

                if num_available > 0:
                    # Random selection instead of importance-based
                    k = min(num_to_add, num_available)
                    perm = torch.randperm(num_available, device=self.device)
                    selected_indices = available_indices[perm[:k]]
                    
                    # Update mask with randomly selected connections
                    self.mask[param_name].view(-1)[selected_indices] = 1.0

                    # Initialize weights for newly added connections
                    param_data = dict(self.model.named_parameters())[param_name].data
                    param_device = param_data.device

                    if param_name in self.init_weights:
                        init_w = self.init_weights[param_name].to(param_device)
                        if init_w.shape == param_data.shape:
                            new_connection_flat_mask = torch.zeros_like(param_data.view(-1), device=param_device)
                            new_connection_flat_mask[selected_indices] = 1.0
                            new_connection_mask = new_connection_flat_mask.reshape(param_data.shape).bool()
                            param_data[new_connection_mask] = init_w[new_connection_mask]
                        else:
                            print(f"Warning: Cannot initialize new connections for {param_name} (shape mismatch).")
                    else:
                        print(f"Warning: Cannot initialize new connections for {param_name} (init_weights missing).")

                # Apply masks to model parameters after growing
                self._apply_masks_to_model()

    # --- Task Transition ---
    def prepare_next_task(self):
        """Prepares the network state for the next task."""
        print(f"\n--- Preparing for Task {self.current_task + 1} ---")
        self.create_masks()
        self.initialize_new_task_weights()
        # Apply masks to model parameters
        self._apply_masks_to_model()
        print(f"--- Ready for Task {self.current_task + 1} ---")
        return True
    
    def save_current_mask(self):
        with torch.no_grad():
            for name in self.mask:
                 current_mask = self.mask[name].to(self.device)
                 if name in self.previous_mask:
                      prev_mask = self.previous_mask[name].to(self.device)
                      self.previous_mask[name] = ((prev_mask + current_mask) > 0).float()
                 else:
                      self.previous_mask[name] = (current_mask > 0).float()

        self.current_task += 1

    def set_evaluation_mask(self):
        """Applies the accumulated mask (union of all task masks) for evaluation."""
        print("Applying accumulated mask (union of all tasks) for evaluation...")
        self.model.eval() # Set model to evaluation mode
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param_device = param.device
                if name in self.previous_mask: # Only apply to parameters managed by CL
                    # Get the accumulated mask (represents union of all learned connections)
                    accumulated_mask = self.previous_mask[name].to(param_device)
                    # Apply the mask: zero out weights that were never used
                    param.data *= accumulated_mask

    def initialize_new_task_weights(self):
        """Initializes weights for the new task's mask."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.mask:
                    current_mask = self.mask[name].to(param.device)
                    if name not in self.previous_mask:
                        self.previous_mask[name] = torch.zeros_like(param.data, device=self.device)
                    prev_mask = self.previous_mask[name].to(param.device)

                    combined_mask = ((prev_mask + current_mask) > 0).float()
                    param.data *= combined_mask
                    new_connections_mask = (current_mask == 1) & (prev_mask == 0)

                    if name in self.init_weights:
                        init_w = self.init_weights[name].to(param.device)
                        if init_w.shape == param.data.shape:
                            param.data[new_connections_mask] = init_w[new_connections_mask]
                        else:
                            print(f"Warning: Shape mismatch for init_weights {name}. Using random init.")
                            stdv = 1. / math.sqrt(param.size(1) if param.dim() > 1 else param.size(0)) if param.numel() > 0 else 0.1
                            param.data[new_connections_mask] = torch.randn_like(param.data[new_connections_mask]) * stdv
                    else:
                        print(f"Warning: init_weights not found for {name}. Using random init.")
                        stdv = 1. / math.sqrt(param.size(1) if param.dim() > 1 else param.size(0)) if param.numel() > 0 else 0.1
                        param.data[new_connections_mask] = torch.randn_like(param.data[new_connections_mask]) * stdv

                    # Fix unselected parameters
                    param.requires_grad = False
                    # Only enable gradients for selected parameters
                    param.requires_grad = True

