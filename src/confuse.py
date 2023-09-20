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
    elif op == 'Conv1x1':
        return Conv(in_c,out_c,1,stride)
    elif op == 'Conv1x1BN':
        return ConvBN(in_c,out_c,1,stride)
    elif op == 'Mbv1Block':
        return Mbv1Block(in_c,out_c,kernel,stride)
    elif op == 'BN':
        return nn.BatchNorm2d(in_c)
    elif op == 'ReLU':
        return nn.ReLU(inplace=True)
    else:
        return None

if __name__ == '__main__':
    op = 'DWConv'
    input=112
    in_c = 32
    out_c = 32
    stride = 1
    k=3
    op_list = ['DWConv',"DWConvBN",'Mbv1Block','Conv1x1','Conv1x1BN','BN','ReLU']
    data_root = "../paper"
    type = '{}_{}_confuse'.format(input,op)
    
    # title
    title = ['op','H', 'in_c', 'out_c', 'stride', 'kernel_size', 'FLOPs', 'Params', 'MACs']
    export_path = '{}/{}'.format(data_root,type)
    make_dirs(export_path)
    csv_path = '{}/data/{}.csv'.format(export_path,type)
    make_dirs('{}/data'.format(export_path))
    with open(csv_path, 'w',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(title)
        for op in op_list:
            model = get_op(op,k,in_c,out_c,stride)
            # model = invert(in_c,out_c,1,1,k)
            tmp_i = (in_c,input,input)
            stat(model, tmp_i)
            model.cuda()
            FLOPs, Params, MACs = measure_model(model, tmp_i)
            print(FLOPs, Params, MACs)
            tmp = [op, input, in_c, out_c, stride, k, FLOPs/(1024*1024), Params/(1024*1024), MACs/(1024*1024)]
            writer.writerow(tmp)
            tmp_i = torch.randn(1, in_c, input, input)
            onnx_export(model, tmp_i.cuda() ,export_path, '{}'.format(op))