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
# from measure import measure_model
import csv
from lib.net_measure import measure_model
def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)



if __name__ == '__main__':
    op = 'DWConv'
    input=112
    in_c = 32
    out_c = 32
    stride = 1
    k=3
    op_list = ['FLOPS','Bandwidth']
    data_root = "../paper"
    type = 'benchmark'
    
    # title
    title = ['op','H', 'in_c', 'out_c', 'stride', 'kernel_size', 'FLOPs', 'Params', 'MACs']
    export_path = '{}/{}'.format(data_root,type)
    make_dirs(export_path)
    csv_path = '{}/data/{}.csv'.format(export_path,type)
    make_dirs('{}/data'.format(export_path))
    with open(csv_path, 'w',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(title)
        op = "Conv"
        in_c = 64
        out_c = 64
        input = 112
        model = Conv(in_c,out_c,7,1)
        
        tmp_i = (in_c,input,input)
        stat(model, tmp_i)
        summary(model.cuda(),tmp_i)
        FLOPs, Params, MACs = measure_model(model, tmp_i)
        print(FLOPs/(1024*1024), Params/(1024*1024), MACs/(1024*1024))
        tmp = [op, input, in_c, out_c, stride, k, FLOPs/(1024*1024), Params/(1024*1024), MACs/(1024*1024)]
        writer.writerow(tmp)
        tmp_i = torch.randn(1, in_c, input, input)
        onnx_export(model, tmp_i.cuda() ,export_path, '{}'.format("Conv_FLOPS"))
        op = "DWConv"
        
        model = DWConv(in_c,out_c,1,1)
        # model = invert(in_c,out_c,1,1,k)
        tmp_i = (in_c,input,input)
        model.cuda()
        FLOPs, Params, MACs = measure_model(model, tmp_i)

        tmp = [op, input, in_c, out_c, stride, k, FLOPs/(1024*1024), Params/(1024*1024), MACs/(1024*1024)]
        writer.writerow(tmp)
        tmp_i = torch.randn(1, in_c, input, input)
        onnx_export(model, tmp_i.cuda() ,export_path, '{}'.format("DWConv_Bandwidth"))