"""
develop a model based on the onnx file in model/model.onnx 

Note:
    - initialize the convolutions layer with uniform xavier
    - initialize the linear layer with a normal distribution (mean=0.0, std=1.0)
    - initialize all biases with zeros
    - use batch norm wherever is relevant
    - use random seed 8
    - use default values for anything unspecified
    
Requirements:
    - Install onnx and onnx2pytorch
    - pip install onnx
    - pip install onnx2pytorch
"""

import numpy as np
import torch
import torch.nn as nn
import onnx
from onnx2pytorch import ConvertModel

torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!

# load the ONNX model
onnx_model = onnx.load('model/model.onnx')

# convert the ONNX model to a PyTorch model
pytorch_model = ConvertModel(onnx_model)

# initialize the layers according to above conditions and add batch normalization layers
for name, module in pytorch_model.named_modules():
    # initialize the convolutions layer with uniform xavier
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data)
        # initialize all biases with zeros
        if module.bias is not None:
            module.bias.data.zero_()
        # add 2D batch norm 
        pytorch_model._modules[name] = nn.Sequential(
            module,
            nn.BatchNorm2d(module.out_channels)
        )
    # initialize the linear layer with a normal distribution (mean=0.0, std=1.0)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight.data, mean=0.0, std=1.0)
        # initialize all biases with zeros
        if module.bias is not None:
            module.bias.data.zero_()
        # add 1D batch norm 
        pytorch_model._modules[name] = nn.Sequential(
            module,
            nn.BatchNorm1d(module.out_features)
        )

# print model defination
print(pytorch_model)

# test the model with a random input
input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
dummy_input = torch.randn(tuple(input_shape), dtype=torch.float32)
output = pytorch_model(dummy_input)
print(output)

