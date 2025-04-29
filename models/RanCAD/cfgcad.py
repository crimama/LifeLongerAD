import copy
import importlib

import torch
import torch.nn as nn

from .criterion import IUFCriterion 

def to_device(input, device="cuda", dtype=None):
    """Transfer data between devidces"""

    if "image" in input:
        input["image"] = input["image"].to(dtype=dtype)

    def transfer(x):
        if torch.is_tensor(x):
            return x.to(device=device)
        elif isinstance(x, list):
            return [transfer(_) for _ in x]
        elif isinstance(x, Mapping):
            return type(x)({k: transfer(v) for k, v in x.items()})
        else:
            return x

    return {k: transfer(v) for k, v in input.items()}

class CFGCAD(nn.Module):
    """Build model from cfg"""

    def __init__(self, net_cfg, criterion_cfg, backbone=None):
        super(CFGCAD, self).__init__()

        self.frozen_layers = []
        for i,cfg_subnet in enumerate(net_cfg):
            mname = cfg_subnet["name"]
            kwargs = cfg_subnet["kwargs"]
            mtype = cfg_subnet["type"]
            if cfg_subnet.get("frozen", False):
                self.frozen_layers.append(mname)
            if cfg_subnet.get("prev", None) is not None:
                prev_module = getattr(self, cfg_subnet["prev"])
                kwargs["inplanes"] = prev_module.get_outplanes()
                kwargs["instrides"] = prev_module.get_outstrides()

            module = self.build(mtype, kwargs)
            self.add_module(mname, module)
            
        self._criterion = IUFCriterion(criterion_cfg)

    def build(self, mtype, kwargs):
        module_name, cls_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)        
        return cls(**kwargs)

    def cuda(self):
        self.device = torch.device("cuda")
        return super(CFGCAD, self).cuda()

    def cpu(self):
        self.device = torch.device("cpu")
        return super(CFGCAD, self).cpu()

    def forward(self, input, **kwargs) -> dict:
        input = copy.copy(input)
        task_id = input.get('clslabel')
        
        if input["image"].device != self.device:
            input = to_device(input, device=self.device)
        for module_num, submodule in enumerate(self.children()):
            if module_num == 2:
                output = submodule(input, task_id)
            else: 
                output = submodule(input)
            input.update(output)
        return output

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self
    
    def criterion(self, Outputs, Inputs, skip:bool = True):
        loss = self._criterion(Outputs, Inputs)
        return loss
