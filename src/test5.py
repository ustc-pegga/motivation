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

M = 1024 * 1024
K = 1024
def get_result(envpath,execute,datasets,device,threads):
    result = os.popen('adb shell "{} && {} {} {} {}"'.format(envpath,execute,datasets,device,threads))
    context = result.read()
    return context


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_model(model_type,kernel_size):
    if model_type == 'mbv1':
        return MobileNet(n_class=25, kernel_size=kernel_size)
    elif model_type == 'mbv2':
        return MobileNetV2(n_class=25, kernel_size=kernel_size)
    elif model_type == 'resnet18':
        pass
    else:
        print("model type error")

def get_input(datasets):
    # return in_c, ,out_c,H
    if datasets == 'imagenet':
        return 3,25,224
    elif datasets == 'cifar10':
        return 3,10,32
    else:
        print("datasets type error")

def get_op(op_type):
    if op_type == 'invert':
        return InvertedResidual
    elif op_type == 'dwconv':
        return DWConv
    else:# op_type == 'conv':
        return Conv


def generate_data_test5_model(model_type,datasets):
    # generate some dw op
    # op type
    # op = 'dwconv'
    # test mode
    type = 'mbv1'
    # 
    times = 1000
    # title
    title = ['op','index','H', 'in_c', 'out_c', 'stride', 'kernel_size', 'FLOPs', 'Params', 'MACs']
    stride = [1,2]
    kernel_size = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]
    export_path = '../data/test5/test5_{}/{}'.format(model_type,datasets)
    make_dirs(export_path)
    csv_path = '../data/test5/test5_{}/{}/test5_{}_{}.csv'.format(model_type,datasets,model_type,datasets)
    with open(csv_path, 'w',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(title)
        in_c, out_c, H = get_input(datasets)
        idx = 0
        input = (in_c,H,H)
        for k in kernel_size:
            
            s = -1 # means model
            model = get_model(model_type,k)
            input = (in_c,H,H)
            FLOPs, Params, MACs = measure_model(model, input)
            tmp = [model_type, idx, H, in_c, out_c, s, k, FLOPs, Params, MACs]
            writer.writerow(tmp)
            input = torch.randn(1, in_c, H, H)
            onnx_export(model, input.cuda() ,export_path, '{}_{}_{}'.format(model_type,"k",str(k)))

def generate_data_test5():
    model_list = ['mbv1','mbv2','resnet18']
    for model in model_list:
        generate_data_test5_model(model,'imagenet')

def generate_data_test5_conv(datasets):
    # generate some dw op
    # op type
    # op = 'dwconv'
    # test mode
    model_type = 'conv'
    type = ''
    # 
    # title
    title = ['op','index','H', 'in_c', 'out_c', 'stride', 'kernel_size', 'FLOPs', 'Params', 'MACs']
    kernel_size = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]
    export_path = '../data/test5/test5_{}/{}'.format(model_type,datasets)
    make_dirs(export_path)
    csv_path = '../data/test5/test5_{}/{}/test5_{}_{}.csv'.format(model_type,datasets,model_type,datasets)
    with open(csv_path, 'w',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(title)
        # in_c, out_c, H = get_input(datasets)
        in_c = 3
        out_c = 32
        H = 112
        s = 2
        idx = 0
        input = (in_c,H,H)
        for k in kernel_size:
            
            s = -1 # means model
            model = get_model(model_type,k)
            input = (in_c,H,H)

            FLOPs, Params, MACs = measure_model(model, input)
            tmp = [type, idx, H, in_c, out_c, s, k, FLOPs, Params, MACs]
            writer.writerow(tmp)
            input = torch.randn(1, in_c, H, H)
            onnx_export(model, input.cuda() ,export_path, '{}_{}_{}'.format(model_type,"k",str(k)))


def generate_data_test5_invert(datasets):
    # generate some dw op
    # op type
    # op = 'dwconv'
    # test mode
    model_type = 'invert'
    type = 'invert'
    # 
    # title
    title = ['op','index','H', 'in_c', 'out_c', 'stride', 'kernel_size', 'FLOPs', 'Params', 'MACs']
    kernel_size = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]
    export_path = '../data/test5/test5_{}/{}'.format(model_type,datasets)
    make_dirs(export_path)
    csv_path = '../data/test5/test5_{}/{}/test5_{}_{}.csv'.format(model_type,datasets,model_type,datasets)
    with open(csv_path, 'w',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(title)
        in_c, out_c, H = get_input(datasets)
        in_c = 16
        out_c = 24
        H = 112
        s = 1
        idx = 0
        expand_ratio = 1
        input = (in_c,H,H)
        for k in kernel_size:
            
            s = -1 # means model
            invert = get_op(model_type)
            input = (in_c,H,H)
            model = invert(in_c,out_c,1,1,k)
            FLOPs, Params, MACs = measure_model(model, input)
            tmp = [type, idx, H, in_c, out_c, s, k, FLOPs, Params, MACs]
            writer.writerow(tmp)
            input = torch.randn(1, in_c, H, H)
            onnx_export(model, input.cuda() ,export_path, '{}_{}_{}'.format(model_type,"k",str(k)))

def generate_data_test5_mbv2(datasets):
    # generate some dw op
    # op type
    # op = 'dwconv'
    # test mode
    model_type = 'mbv2'
    type = 'mbv2'
    # 
    # title
    title = ['op','index','H', 'in_c', 'out_c', 'stride', 'kernel_size', 'FLOPs', 'Params', 'MACs']
    kernel_size = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]
    export_path = '../data/test5/test5_{}/{}'.format(model_type,datasets)
    make_dirs(export_path)
    csv_path = '../data/test5/test5_{}/{}/test5_{}_{}.csv'.format(model_type,datasets,model_type,datasets)
    with open(csv_path, 'w',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(title)
        in_c, out_c, H = get_input(datasets)
        in_c = 3
        out_c = 25
        H = 224
        s = 224
        idx = 0
        expand_ratio = 1
        input = (in_c,H,H)
        for k in kernel_size:
            
            s = -1 # means model
            model = get_model(model_type,k)
            input = (in_c,H,H)
            # model = invert(in_c,out_c,1,1,k)
            FLOPs, Params, MACs = measure_model(model, input)
            tmp = [type, idx, H, in_c, out_c, s, k, FLOPs, Params, MACs]
            writer.writerow(tmp)
            input = torch.randn(1, in_c, H, H)
            onnx_export(model, input.cuda() ,export_path, '{}_{}_{}'.format(model_type,"k",str(k)))



def show_model(model_type):
    model = get_model(model_type,11)
    input = torch.randn(1, 3, 224, 224)
    input = (3,224,224)
    FLOPs, Params, MACs = measure_model(model, input)
    summary(model.cuda(), (3, 224, 224))
    print('FLOPs: %.2fM, Params: %.2fM, MACs: %.2fM' % (FLOPs / M, Params / M, MACs / M))

if __name__ == "__main__":
    generate_data_test5_mbv2('imagenet')
    # show_model('mbv1')