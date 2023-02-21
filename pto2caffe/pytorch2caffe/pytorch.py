import torch
from torch import nn
from torch import Tensor

from typing import Dict, Iterable, Callable

def shape2size(shape:list):
    size = 1
    for dim in shape:
        size *= dim

    return size

def get_output(model, input_tensor, layer_name):
#    device = torch.device('cpu')#torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    input_tensor = torch.from_numpy(input_tensor).to('cpu')
    if not model.status.forwarded:
        model.pnnx.model_forward(input_tensor)
        model.setForwarded()

    for op in model.operators:
        if op.name == layer_name:
            return model.pnnx.get_ops_output(layer_name, shape2size(op.outputs_shape[0])) 
