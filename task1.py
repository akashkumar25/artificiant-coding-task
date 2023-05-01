"""
Write a code using pytorch to replicate a grouped 2D convolution layer based on the original 2D convolution. 

The common way of using grouped 2D convolution layer in Pytorch is to use 
torch.nn.Conv2d(groups=n), where n is the number of groups.

However, it is possible to use a stack of n torch.nn.Conv2d(groups=1) to replicate the same
result. The wights must be copied and be split between the convs in the stack.

You can use:
    - use default values for anything unspecified  
    - all available functions in NumPy and Pytorch
    - the custom layer must be able to take all parameters of the original nn.Conv2d 
"""

import numpy as np
import torch
import torch.nn as nn


torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!

# random input (batch, channels, height, width)
x = torch.randn(2, 64, 100, 100)

# original 2d convolution
grouped_layer = nn.Conv2d(64, 128, 3, stride=1, padding=1, groups=16, bias=True)

# weights and bias
w_torch = grouped_layer.weight
b_torch = grouped_layer.bias

y = grouped_layer(x)

# now write your custom layer
class CustomGroupedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super().__init__()
        
        # save the parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.bias = bias
        
        # calculate the number of input channels and output channels for each group
        self.in_channels_per_group = in_channels // groups
        self.out_channels_per_group = out_channels // groups
        
        # create a stack of 1-grouped convolutions
        self.conv_stack = nn.ModuleList([
            nn.Conv2d(self.in_channels_per_group, self.out_channels_per_group, self.kernel_size, stride=self.stride,
                      padding=self.padding, groups=1, bias=self.bias)
            for _ in range(self.groups)
        ])
        
        # copy and split the weights and bais from the original 2d convolution layer
        for idx, conv in enumerate(self.conv_stack):
            start = idx * self.out_channels_per_group
            end = (idx + 1) * self.out_channels_per_group
            conv.weight.data = w_torch[start:end]
            if self.bias:
                conv.bias.data = b_torch[start:end]


    def forward(self, x):
        
        # split the input and pass through the convolutions
        x_split = torch.split(x, self.in_channels_per_group, dim=1)
        out = [conv(x_split[i]) for i, conv in enumerate(self.conv_stack)]
        # concatenate the outputs
        out = torch.cat(out, dim=1)
        return out

# initialize the custom grouped convolution layer with the same parameters as the original convolution
custom_layer = CustomGroupedConv2D(64, 128, 3, stride=1, padding=1, groups=16, bias=True)

# test the custom layer
y_custom = custom_layer(x)

# check that the output of the custom layer is equal to the output of the original layer
# This would allow for a relative tolerance of 0.1% and an absolute tolerance of 1e-5 when comparing the outputs.
print(torch.allclose(y_custom, y, rtol=1e-3, atol=1e-5))

# the output of CustomGroupedConv2D(x) must be equal to grouped_layer(x)
# Output: True
