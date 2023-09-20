
from torchstat import stat
from torchsummary import summary
import os
import sys
import csv
import random
from model.op import * 
from measure import measure_model
from ms_export import onnx_export
import argparse


import os
import time
import argparse
import shutil
import math
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.model import *
from pruning import *
from tran import tran 
from ms_export import onnx_export
def parse_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--op', default='DWConv', type=str, help='name of the model to train')
    parser.add_argument('--input', default='128', type=int, help='name of the dataset to train')
    parser.add_argument('--kernel', default='3', type=int, help='kernel_size')
    parser.add_argument('--cin', default=None, type=int, help='kernel_size')
    parser.add_argument('--cout', default=None, type=int, help='pruning')
    parser.add_argument('--stride',default=1, type=int,help='stride')
    parser.add_argument('--name', default="", type=str, help='pruning')
    parser.add_argument('--type', default=None, type=str, help='pruning')
    parser.add_argument('--data_root', default=None, type=str, help='root')
    return parser.parse_args()


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




if __name__ == '__main__':
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    op = args.op
    input = args.input
    kernel = args.kernel # args.kernel
    in_c = args.cin
    out_c = args.cout
    stride = args.stride
    kernel = args.kernel
    data_root = args.data_root
    print('=> Preparing data..')
    net = get_op(op,kernel,in_c,out_c,stride)
    
    export_path = os.path.join(data_root,args.type)
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    print(export_path)
    onnx_export(net, torch.randn(1,in_c,input,input) ,export_path, '{}_{}_{}_{}_{}_{}'.format(op,input,in_c,out_c,kernel,stride))
    flops,params,macs = measure_model(net,(in_c,input,input))
    # print(model,dataset,args.name,flops,params,macs)
    # print(net)


