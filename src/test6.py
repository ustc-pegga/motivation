import torch.nn as nn
import torch
import random
from measure import measure_model
from torchstat import stat
from torchsummary import summary
import os
import sys
import csv
import random
from model.op import * 
from measure import measure_model
from ms_export import onnx_export
from model.model import * 
from pruning import *
def save_model(model_type,dataset,ratio,pth_path,export_path,name):
    if model_type == "mbv1":
        net = get_mbv1(dataset,ratio,pth_path)
    else:
        net = get_mbv2(dataset,ratio,pth_path)
    input = torch.randn(1, 3, 224, 224)
    onnx_export(net, input ,export_path, name)
    return net
def get_mbv1(dataset,ratio,pruned_path):
    if dataset == "imagenet-25":
        # net =  MobileNet(n_class=25, kernel_size=3)
        net =  MobileNet(n_class=25, kernel_list = [3, 5, 3, 5, 3, 5, 3, 3, 3, 3, 3, 5, 3])
    else:# cifar10
        model = MobileNet(n_class=10)
        net = MobileNet(n_class=25)

    # ckpt_path = "/home/hujie/code/motivation/checkpoint/mobilenet_k3_imagenet25_300-run7/ckpt.best.pth.tar"
    ckpt_path = "/home/hujie/code/motivation/kernel/mbv1_in3.pth.tar"
    
    # ckpt_path = "/home/hujie/code/amc/logs/mobilenet_imagenet-10_finetune-run81/ckpt.best.pth.tar"
    # ckpt_path = "/home/hujie/code/amc/logs/mobilenet_imagenet-10_finetune-run88/ckpt.best.pth.tar"
    if ckpt_path is not None:  # assigned checkpoint path to resume from
        print('=> Resuming from checkpoint..')
        checkpoint = torch.load(ckpt_path)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_dict = {k: v for k, v in sd.items() if net.state_dict()[k].numel() == v.numel()}
        # sd.update(new_dict)
        missing_keys, unexpected_keys = net.load_state_dict(new_dict, strict=False)
    net = mbv1_pruning(net,ratio,input=(224,224))
    # net.state_dict().update(torch.load(pruned_path))
    return net

def get_mbv2(dataset,ratio,pruned_path):
    if dataset == "imagenet-25":
        # net =  MobileNetV2(n_class=25, kernel_size=3)
        net = MobileNetV2(n_class=25,kernel_list=[3, 5, 3, 5, 3, 3, 5, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3])
    else:# cifar10
        model = MobileNet(n_class=10)
        net = MobileNet(n_class=25)

    ckpt_path = "/home/hujie/code/motivation/kernel/ckpt.best.pth.tar"
    # ckpt_path = "/home/hujie/code/amc/logs/mobilenet_imagenet-10_finetune-run81/ckpt.best.pth.tar"
    # ckpt_path = "/home/hujie/code/amc/logs/mobilenet_imagenet-10_finetune-run88/ckpt.best.pth.tar"
    if ckpt_path is not None:  # assigned checkpoint path to resume from
        print('=> Resuming from checkpoint..')
        checkpoint = torch.load(ckpt_path)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_dict = {k: v for k, v in sd.items() if net.state_dict()[k].numel() == v.numel()}
        # sd.update(new_dict)
        missing_keys, unexpected_keys = net.load_state_dict(new_dict, strict=False)
    net = mbv2_pruning(net,ratio,input=(224,224))
    # net.state_dict().update(torch.load(pruned_path))
    return net

if __name__ == "__main__":
    intensity_path = "../mobilenet/mbv2_imagenet25.pth.tar"
    export_path = "/home/hujie/code/motivation/data/test6/mbv2/imagenet"
    # intensity 3
    ratio = [1.0, 0.75, 0.8125, 0.78125, 0.75, 0.75, 0.765625, 0.7421875, 0.7265625, 0.7109375, 0.703125, 0.6796875, 0.671875, 0.4609375, 0.203125]
    # intensity 20
    ratio =  [1.0, 0.75, 0.8125, 0.78125, 0.8125, 0.796875, 0.8125, 0.7578125, 0.7421875, 0.703125, 0.671875, 0.6328125, 0.640625, 0.3671875, 0.203125]

    # mbv2 in3
    ratio = [1.0, 1.0, 1.0, 0.9166666666666666, 1.0, 0.7222222222222222, 1.0, 0.6111111111111112, 1.0, 0.5416666666666666, 1.0, 0.5208333333333334, 1.0, 0.5416666666666666, 0.875, 0.5104166666666666, 0.875, 0.5104166666666666, 0.875, 0.5104166666666666, 0.875, 0.5104166666666666, 0.625, 0.5, 0.625, 0.5138888888888888, 0.625, 0.5486111111111112, 0.625, 0.5, 0.65, 0.49166666666666664, 0.65, 0.37916666666666665, 0.275, 1.0]

    # mbv2 in20
    # ratio = [1.0, 0.625, 0.75, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6944444444444444, 0.75, 0.7083333333333334, 0.75, 0.6666666666666666, 0.75, 0.6875, 0.875, 0.7083333333333334, 0.875, 0.5833333333333334, 0.875, 0.7395833333333334, 0.875, 0.6770833333333334, 0.875, 0.6319444444444444, 0.875, 0.5833333333333334, 0.875, 0.7083333333333334, 0.85, 0.6291666666666667, 0.75, 0.625, 0.875, 0.7208333333333333, 0.4375, 0.2]

    # ratio = [1.0, 0.75, 0.75, 0.75, 0.8333333333333334, 0.6388888888888888, 0.8333333333333334, 0.6388888888888888, 0.875, 0.625, 0.875, 0.6458333333333334, 0.875, 0.6458333333333334, 0.8125, 0.65625, 0.8125, 0.65625, 0.8125, 0.65625, 0.8125, 0.65625, 0.6666666666666666, 0.5902777777777778, 0.6666666666666666, 0.6388888888888888, 0.6666666666666666, 0.6666666666666666, 0.65, 0.6625, 0.675, 0.6875, 0.675, 0.7041666666666667, 0.525, 0.590625]
    # ratio = [1.0, 0.875, 0.875, 0.84375, 0.78125, 0.75, 0.71875, 0.7265625, 0.71875, 0.6875, 0.671875, 0.65625, 0.546875, 0.44140625, 0.30859375]
    
    #mbv1 60%
    # ratio = [1.0, 0.5, 0.625, 0.625, 0.65625, 0.640625, 0.65625, 0.6640625, 0.671875, 0.671875, 0.671875, 0.671875, 0.6640625, 0.421875, 0.203125]

    #mbv2 kernel_pruning 
    ratio = [1.0, 0.5, 0.75, 0.5833333333333334, 0.5, 0.5, 0.6666666666666666, 0.5277777777777778, 0.75, 0.5625, 0.75, 0.5625, 0.75, 0.625, 0.625, 0.65625, 0.75, 0.6666666666666666, 0.75, 0.59375, 0.75, 0.6979166666666666, 0.75, 0.6875, 0.875, 0.7638888888888888, 0.7916666666666666, 0.75, 0.775, 0.6958333333333333, 0.775, 0.6916666666666667, 0.75, 0.7125, 0.65, 0.553125]
    net = save_model("mbv2","imagenet-25",ratio,intensity_path,export_path,"mbv2_imagenet-25_kernel_pruning")
    print(net)
    # input = torch.randn(1, 3, 224, 224)
    # # net = MobileNetV2()
    # # onnx_export(net, input ,export_path, "imagenet-25_origin")   [3, 5, 3, 5, 3, 5, 3, 3, 3, 3, 3, 5, 3]
    # kernel_list = [3, 5, 3, 5, 3, 3, 5, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3]
    # net = MobileNet(n_class=25)
    # onnx_export(net, input ,export_path, "mbv1_imagenet-25_origin")
