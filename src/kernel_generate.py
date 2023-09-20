import os
import time
import argparse
import shutil
import math
import gc
import numpy as np
from model.model import *
from pruning import *
from tran import tran 
from ms_export import onnx_export
from model.op import * 
from lib.net_measure import measure_model
import csv

def get_cmd(device,op,input,cin,cout,stride,kernel):
    cmd = "python dwkernel.py \
        --type kernel \
        --data_root ../paper \
        --device {} \
        --op {} \
        --input {} \
        --cin {} \
        --cout {} \
        --stride {} \
        --kernel {} ".format(device,op,input,cin,cout,stride,kernel)
def execute(cmd):
    result = os.popen('adb shell {}'.format(cmd))
    context = result.read()
    return context

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_op(op,kernel,in_c,out_c,stride):
    if op == "DWConv":
        return DWConv(in_c,out_c,kernel,stride)
    elif op == "Conv":
        return Conv(in_c,out_c,kernel,stride)
    elif op == 'DWConvBN':
        return DWConvBN(in_c,out_c,kernel,stride)
    elif op == 'ConvBN':
        return ConvBN(in_c,out_c,kernel,stride)
    else:
        return None

def get_intensity(h,kernel,stride,in_c,out_c,groups):
    in_h = h
    in_w = h
    padding = (kernel - 1) // 2
    out_h = int((in_h + 2 * padding - kernel) /
                stride + 1)
    out_w = int((in_h + 2 * padding - kernel) /
                stride + 1)
    flops = in_c * out_c * kernel *  \
                kernel * out_h * out_w / groups * 1
    params = kernel * kernel * in_c * out_c / groups
    macs  = in_c * in_h * in_w + out_c * out_h * out_w + \
        params
    params *= 4
    macs *= 4

    flops += 3 * out_h * out_h * out_c
    return flops,params,macs

if __name__ == '__main__':
    op = 'ConvBN'
    input=112
    in_c = 32
    out_c = 32
    stride = 1
    kernel_list = [1,3,5,7,9,11,13,15,17]
    data_root = "../paper"
    type = '{}_{}_intensity'.format(input,op)
    
    # title
    # title = ['op','H', 'in_c', 'out_c', 'stride', 'kernel_size', 'FLOPs', 'Params', 'MACs']
    # export_path = '{}/{}'.format(data_root,type)
    # make_dirs(export_path)
    # csv_path = '{}/data/{}.csv'.format(export_path,type)
    # make_dirs('{}/data'.format(export_path))
    # with open(csv_path, 'w',encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(title)
    #     for k in kernel_list:
    #         if k > input:
    #             continue
    #         model = get_op(op,k,in_c,out_c,stride)
    #         # model = invert(in_c,out_c,1,1,k)
    #         model.cuda()
    #         tmp_i = (in_c,input,input)
    #         if "DW" in type:
    #             groups = in_c
    #         else:
    #             groups = 1
    #         FLOPs, Params, MACs = get_intensity(input,k,stride,in_c,out_c,groups)
    #         # FLOPs, Params, MACs = measure_model(model, tmp_i)

    #         tmp = [op, input, in_c, out_c, stride, k, FLOPs/(1024*1024), Params/(1024*1024), MACs/(1024*1024)]
    #         writer.writerow(tmp)
    #         tmp_i = torch.randn(1, in_c, input, input)
    #         onnx_export(model, tmp_i.cuda() ,export_path, '{}_{}_{}'.format(op,"k",str(k)))
    
