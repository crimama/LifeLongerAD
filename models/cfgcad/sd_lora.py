import math
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
import torch.nn.utils as utils

class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x) + self.w_b(self.w_a(x))
        return x


class ParameterWrapper(nn.Module):
    def __init__(self, param):
        super(ParameterWrapper, self).__init__()
        self.param = param
    
    def forward(self, x):
        return x * self.param


class _LoRA_qkv(nn.Module):
    """
    LoRA implementation for fused QKV projection in attention layers
    """
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*dim
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv


class _LoRA_qkv_timm_train(nn.Module):
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v,
        task_id, saved_A, saved_B, t_layer_i, rank, scaling_factor, scaling_factor_prev, eval1=False):
        super().__init__()
        self.linear_a_q = linear_a_q.cuda()
        self.linear_b_q = linear_b_q.cuda()
        self.linear_a_v = linear_a_v.cuda()
        self.linear_b_v = linear_b_v.cuda()

        self.scaling_factor = scaling_factor.cuda()
        self.scaling_factor_prev = scaling_factor_prev.cuda()

        self.task_id = task_id
        self.qkv = qkv
        self.dim = qkv.in_features
        self.saved_A = saved_A
        self.saved_B = saved_B
        self.t_layer_i = t_layer_i
        self.rank = rank
        self.eval = eval1

    def forward(self, x):
        w_a_linear_q = nn.Linear(self.dim, self.rank, bias=False)
        w_b_linear_q = nn.Linear(self.rank, self.dim, bias=False)
        w_a_linear_v = nn.Linear(self.dim, self.rank, bias=False)
        w_b_linear_v = nn.Linear(self.rank, self.dim, bias=False)
         
        new_q, new_v = 0, 0
        for i in range(self.task_id):
            saved_A_i, saved_B_i = self.saved_A['saved_A_'+str(i)], self.saved_B['saved_B_'+str(i)]
            Q, V = list(enumerate(zip(saved_A_i,saved_B_i)))[self.t_layer_i*2: self.t_layer_i*2+2]
            _, (A_q, B_q) = Q
            _, (A_v, B_v) = V

            w_a_linear_q.weight = Parameter(A_q.weight)
            w_a_linear_q.weight.requires_grad = False 
            w_a_linear_q.to(x.device)
            w_b_linear_q.weight = Parameter(B_q.weight)
            w_b_linear_q.weight.requires_grad = False 
            w_b_linear_q.to(x.device)
            w_a_linear_v.weight = Parameter(A_v.weight)
            w_a_linear_v.weight.requires_grad = False 
            w_a_linear_v.to(x.device)
            w_b_linear_v.weight = Parameter(B_v.weight)
            w_b_linear_v.weight.requires_grad = False  
            w_b_linear_v.to(x.device)

            if i == 0:
                new_q = self.scaling_factor_prev[i](w_b_linear_q(w_a_linear_q(x))/(torch.norm(w_b_linear_q.weight)*torch.norm(w_a_linear_q.weight)))
                new_v = self.scaling_factor_prev[i](w_b_linear_v(w_a_linear_v(x))/(torch.norm(w_b_linear_v.weight)*torch.norm(w_a_linear_v.weight)))
            else:
                new_q += self.scaling_factor_prev[i](w_b_linear_q(w_a_linear_q(x))/(torch.norm(w_b_linear_q.weight)*torch.norm(w_a_linear_q.weight)))
                new_v += self.scaling_factor_prev[i](w_b_linear_v(w_a_linear_v(x))/(torch.norm(w_b_linear_v.weight)*torch.norm(w_a_linear_v.weight)))

        new_q += self.scaling_factor[0](self.linear_b_q(self.linear_a_q(x)))
        new_v += self.scaling_factor[0](self.linear_b_v(self.linear_a_v(x)))
        qkv = self.qkv(x) 
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv


class _LoRA_qkv_timm_eval(nn.Module):
    def __init__(self, task_id, qkv: nn.Module, saved_A, saved_B, t_layer_i, rank, scaling_factor, scaling_factor_prev, save_file):
        super().__init__()
        self.task_id = task_id
        self.qkv = qkv
        self.dim = qkv.in_features
        self.saved_A = saved_A
        self.saved_B = saved_B
        self.t_layer_i = t_layer_i
        self.rank = rank
        self.save_file = save_file
        self.scaling_factor = scaling_factor.cuda()
        self.scaling_factor_prev = scaling_factor_prev.cuda()

    def forward(self, x):
        new_q, new_v = 0, 0

        w_a_linear_q = nn.Linear(self.dim, self.rank, bias=False)
        w_b_linear_q = nn.Linear(self.rank, self.dim, bias=False)
        w_a_linear_v = nn.Linear(self.dim, self.rank, bias=False)
        w_b_linear_v = nn.Linear(self.rank, self.dim, bias=False)

        file_path = self.save_file+'scaling_factor'+str(self.task_id-1)+'.pt'
        scaling_param = torch.load(file_path)
        for i in range(self.task_id):
            saved_A_i, saved_B_i = self.saved_A['saved_A_'+str(i)], self.saved_B['saved_B_'+str(i)]
            Q, V = list(enumerate(zip(saved_A_i,saved_B_i)))[self.t_layer_i*2: self.t_layer_i*2+2]
            _, (A_q, B_q) = Q
            _, (A_v, B_v) = V

            w_a_linear_q.weight = Parameter(A_q.weight)
            w_b_linear_q.weight = Parameter(B_q.weight)
            w_a_linear_v.weight = Parameter(A_v.weight)
            w_b_linear_v.weight = Parameter(B_v.weight)

            if i == 0:
                new_q = self.scaling_factor_prev[i](w_b_linear_q(w_a_linear_q(x))/(torch.norm(w_b_linear_q.weight)*torch.norm(w_a_linear_q.weight)))
                new_v = self.scaling_factor_prev[i](w_b_linear_v(w_a_linear_v(x))/(torch.norm(w_b_linear_v.weight)*torch.norm(w_a_linear_v.weight)))
            else:
                new_q += self.scaling_factor_prev[i](w_b_linear_q(w_a_linear_q(x))/(torch.norm(w_b_linear_q.weight)*torch.norm(w_a_linear_q.weight)))
                new_v += self.scaling_factor_prev[i](w_b_linear_v(w_a_linear_v(x))/(torch.norm(w_b_linear_v.weight)*torch.norm(w_a_linear_v.weight)))

        new_q = self.scaling_factor[0](w_b_linear_q(w_a_linear_q(x)))
        new_v = self.scaling_factor[0](w_b_linear_v(w_a_linear_v(x)))
 
        qkv = self.qkv(x) 
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv


class LoRA(nn.Module):
    """Applies low-rank adaptation to a model.
    Args:
        model: base model to apply LoRA to
        r: rank of LoRA
        lora_layer: which layers to apply LoRA to
        filepath: path to save/load LoRA weights
        eval: whether in evaluation mode
    """
    def __init__(self, model, r: int, filepath='./', lora_layer=None, eval=False, increment=10, index=True, cur_task_index=None):
        super(LoRA, self).__init__()

        assert r > 0
        self.rank = r
        self.base_model = copy.deepcopy(model)
        
        if not eval:
            self.save_file = filepath
            self.increment = increment

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            # Default to all layers - should be populated based on model architecture
            self.lora_layer = []
        
        # Storage for LoRA weights
        self.w_As = []
        self.w_Bs = []
        
        if index:
            self.task_id, self.cur_id = 0, 0
        
        if cur_task_index is not None:
            self.task_id = cur_task_index

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in model.parameters():
            param.requires_grad = False

        # Load saved LoRA parameters if available
        saved_lora_A, saved_lora_B = {}, {}
        for i in range(self.task_id):
            file_path = self.save_file+'lora_w_a_'+str(i)+'.pt'
            saved_lora_A['saved_A_'+str(i)] = torch.load(file_path)
            file_path = self.save_file+'lora_w_b_'+str(i)+'.pt'
            saved_lora_B['saved_B_'+str(i)] = torch.load(file_path)

        # Initialize scaling factors
        scaling_factor = nn.Parameter(torch.Tensor([0.8]))
        self.wrapped_param = nn.ModuleList([ParameterWrapper(scaling_factor)])
        self.wrapped_param_prev = nn.ModuleList([ParameterWrapper(nn.Parameter(torch.Tensor([0.8]))) for _ in range(20)])

        # Apply LoRA to the model - this method should be implemented for specific model architectures
        self.apply_lora(model, saved_lora_A, saved_lora_B, eval)
        
        self.reset_parameters()
        self.lora_model = model

    def apply_lora(self, model, saved_lora_A, saved_lora_B, eval):
        # This method should be implemented based on the specific model architecture
        # Example implementation for attention layers with QKV:
        """
        for t_layer_i, layer in enumerate(model.layers):
            if t_layer_i not in self.lora_layer:
                continue
                
            w_qkv_linear = layer.self_attn.qkv
            dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(dim, self.rank, bias=False)
            w_b_linear_q = nn.Linear(self.rank, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, self.rank, bias=False)
            w_b_linear_v = nn.Linear(self.rank, dim, bias=False)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            if not eval:
                layer.self_attn.qkv = _LoRA_qkv_timm_train(
                    w_qkv_linear, w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v, 
                    self.task_id, saved_lora_A, saved_lora_B, t_layer_i, self.rank, 
                    self.wrapped_param, self.wrapped_param_prev, eval1=False
                )
            else:
                layer.self_attn.qkv = _LoRA_qkv_timm_eval(
                    self.task_id, w_qkv_linear, saved_lora_A, saved_lora_B, 
                    t_layer_i, self.rank, self.wrapped_param, self.wrapped_param_prev, self.save_file
                )
        """
        pass

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def save_wrap_param(self, filename):
        if self.task_id == 1:   
            scaling_param = torch.zeros(20, 20)
        else:
            scaling_param = torch.load(filename + 'scaling_factor'+str(self.task_id-2)+'.pt')
        i = self.task_id-1
        for j in range(i+1):
            if j == i:
                scaling_param[i][j] = self.wrapped_param[0].param.clone()
            else:
                scaling_param[i][j] = self.wrapped_param_prev[j].param.clone()  
        torch.save(scaling_param, filename + 'scaling_factor'+str(self.task_id-1)+'.pt')
        
    def save_lora_parameters(self, filename: str, task_id) -> None:
        self.task_id += 1
        torch.save(self.w_As, filename + 'lora_w_a_'+str(task_id)+'.pt')
        torch.save(self.w_Bs, filename + 'lora_w_b_'+str(task_id)+'.pt')

    def compute_ortho_loss(self):
        loss = torch.tensor(0).float().cuda()
        for i in range(self.task_id):
            file_path = self.save_file+'lora_w_a_'+str(i)+'.pt'
            if os.path.exists(file_path):
                w_As = torch.load(file_path)
                num_layer = len(self.w_As)
                for j in range(num_layer):
                    temp = torch.matmul(w_As[j].weight.to(self.w_As[j].weight.device), self.w_As[j].weight.t())
                    temp = torch.sum(torch.square(temp))
                    loss = loss.to(self.w_As[j].weight.device)
                    loss += temp
        return loss
    
    def forward(self, x: Tensor, loss=False, eval=False) -> Tensor:
        if eval:
            # Reset model for evaluation
            return self.lora_model(x)
        elif loss:
            loss = self.compute_ortho_loss()
            return self.lora_model(x), loss
        else:
            return self.lora_model(x)