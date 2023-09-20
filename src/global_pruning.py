import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy as np 
import time
from torchstat import stat
from torchsummary import summary
from model.model import *



model1 = MobileNetV2(n_class=1000)
model2 = MobileNetV2(n_class=1000,kernel_size=5)

example_inputs = torch.randn(1, 3, 224, 224)

# 1. Importance criterion
imp = tp.importance.MagnitudeImportance(p=2) # or GroupNormImportance(p=2), GroupHessianImportance(), etc.

# 2. Initialize a pruner with the model and the importance criterion
ignored_layers = []
for m in model1.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
        ignored_layers.append(m) # DO NOT prune the final classifier!

pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
    model1,
    example_inputs,
    importance=imp,
    ch_sparsity=0.5, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    # ch_sparsity_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized sparsity for layers or blocks
    ignored_layers=ignored_layers,
    global_pruning=True
)
macs1, params1 = tp.utils.count_ops_and_params(model1, example_inputs)
macs2, params2 = tp.utils.count_ops_and_params(model2, example_inputs)

a = macs2/macs1
print(a)
pruner.step()
macs, nparams = tp.utils.count_ops_and_params(model1, example_inputs)

print(macs,nparams)
ignored_layers = []
for m in model2.modules():
    if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
        ignored_layers.append(m) # DO NOT prune the final classifier!
pruner = tp.pruner.MetaPruner( # We can always choose MetaPruner if sparse training is not required.
    model2,
    example_inputs,
    importance=imp,
    ch_sparsity=float(0.5*a), # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    # ch_sparsity_dict = {model.conv1: 0.2, model.layer2: 0.8}, # customized sparsity for layers or blocks
    ignored_layers=ignored_layers,
    global_pruning=True
)
pruner.step()
macs, nparams = tp.utils.count_ops_and_params(model2, example_inputs)
print(macs,nparams)