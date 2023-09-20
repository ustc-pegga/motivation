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


M = 1024 * 1024
K = 1024
def get_result(envpath,execute,datasets,device,threads):
    result = os.popen('adb shell "{} && {} {} {} {}"'.format(envpath,execute,datasets,device,threads))
    context = result.read()
    return context


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _MACs_limit(op):
    limit = 20 * M
    if 'dwconv' in op:
        limit = 5 * M
    else:
        limit = 10 * M
    return limit
        

def generate_data_test1_random(op):
    # generate some dw op
    # op type
    # op = 'dwconv'
    # test mode
    type = 'random'
    # 
    times = 1000
    # title
    title = ['type','index','H', 'in_c', 'out_c', 'stride', 'kernel_size', 'FLOPs', 'Params', 'MACs']
    input_size = [224,112,56,28,14,7]
    in_channels = list(range(16,1024+1,4))
    out_channels = list(range(16,1024+1,4))
    stride = [1,2]
    kernel_size = [1,3,5,7]

    csv_path = '../data/test1/test1_{}.csv'.format(op)
    export_path = '../data/test1/test1_{}'.format(op)
    make_dirs(export_path)
    with open(csv_path, 'w',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(title)

        for i in range(times):
            print("times:",i)
            MACs_limit = _MACs_limit(op)
            MACs = 50 * M
            while(MACs > MACs_limit):
                H = random.choice(input_size)
                in_c = random.choice(in_channels)
                out_c = random.choice(out_channels)
                s = random.choice(stride)
                k = random.choice(kernel_size)
                input = (in_c, H, H)
                if op == 'dwconv':
                    model = DWConv(in_c, in_c, s, k)
                elif op == 'dwconvbn':
                    model = DWConvBN(in_c, out_c, s, k)
                elif op == 'conv':
                    model = Conv(in_c, out_c, s, k)
                elif op == 'convbn':
                    model = ConvBN(in_c, out_c, s, k)
                FLOPs, Params, MACs = measure_model(model, input)
            tmp = [op, i, H, in_c, out_c, s, k, FLOPs, Params, MACs]
            writer.writerow(tmp)
            input = torch.randn(1, in_c, H, H)
            onnx_export(model, input.cuda() ,export_path, '{}_{}'.format(op,str(i)))

def generate_data_test1():
    ops = ['dwconv','dwconvbn','conv','convbn']
    for op in ops:
        generate_data_test1_random(op)


if __name__ == "__main__":
    generate_data_test1_random("convbn")

# def test1(op):


