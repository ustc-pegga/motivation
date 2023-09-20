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
from lib.utils import * 
import matplotlib.pyplot as plt


idx = 1

def new_forward(m):
    def lambda_forward(x):
        m.input_feat = x.clone()
        measure_layer_for_pruning(m, x)
        y = m.old_forward(x)
        m.output_feat = y.clone()
        return y

    return lambda_forward

def get_intensity(layer,kernel):
    in_h = layer.in_h
    in_w = layer.in_h
    out_h = int((in_h + 2 * layer.padding[0] - kernel) /
                layer.stride[0] + 1)
    out_w = int((in_h + 2 * layer.padding[1] - kernel) /
                layer.stride[1] + 1)
    flops = layer.in_channels * layer.out_channels * kernel *  \
                kernel * out_h * out_w / layer.groups * 1
    params = kernel * kernel * layer.in_channels * layer.out_channels / layer.groups
    macs  = layer.in_channels * in_h * in_w + layer.out_channels * out_h * out_w + \
        params
    params *= 4
    macs *= 4

    flops += 3 * out_h * out_h * layer.out_channels
    return float(flops/macs)

def get_macs(layer,kernel):
    in_h = layer.in_h
    in_w = layer.in_h
    out_h = int((in_h + 2 * layer.padding[0] - kernel) /
                layer.stride[0] + 1)
    out_w = int((in_h + 2 * layer.padding[1] - kernel) /
                layer.stride[1] + 1)
    flops = layer.in_channels * layer.out_channels * kernel *  \
                kernel * out_h * out_w / layer.groups * 1
    params = kernel * kernel * layer.in_channels * layer.out_channels / layer.groups
    macs  = layer.in_channels * in_h * in_w + layer.out_channels * out_h * out_w + \
        params
    macs *= 4

    return macs

if __name__ == '__main__':

    cpu_intensity = 12.12
    gpu_intensity = 54.3
    npu_intensity = 69.0
    intensity = {"cpu":cpu_intensity,"gpu":gpu_intensity,"npu":npu_intensity}
    device_list = ['cpu','gpu','npu','origin']
    input_size = 56
    n_class = 100

    for device in device_list:
        net = MobileNet(n_class=n_class)
        if device == 'origin':
            net.eval()
            onnx_export(net.cuda(), tmp_i.cuda() ,export_path, 'mbv1_{}_{}'.format(input_size,device))
            continue
        model = list(net.modules())
        idx = 1
        for idx in range(len(model)):
            m = model[idx]
            m.old_forward = m.forward
            # print(type(new_forward(m)))
            m.forward = new_forward(m)

        # stat(net,(3,32,32))
        input = torch.randn(1,3,input_size,input_size)
        net(input)
        kernel_list = []
        DW = []
        conv = []
        all = []
        idx = 0
        for i in range(len(model)):
            m = model[i]
            kernel = 7
            if type(m) == nn.Conv2d and m.groups == m.in_channels:
                # while(get_intensity(m,kernel) < intensity[device] and kernel < m.in_h and kernel < 13 and get_macs(m,kernel)/m.macs < 1.1):
                
                #     kernel+=2
                print(get_intensity(m,kernel-2))
                kernel_list.append(max(kernel-2,3))
                idx+=1
            elif type(m) == nn.Conv2d:
                pass
                # print(get_intensity(m,1))
            else:
                pass
        print(kernel_list)
        export_path = '{}/{}'.format("../paper","device")
        if not os.path.exists(export_path):
            os.makedirs(export_path)
        tmp_i = torch.randn(1,3,input_size, input_size)
        net = MobileNet(n_class=n_class,kernel_list=kernel_list)
        onnx_export(net.cuda(), tmp_i.cuda() ,export_path, 'mbv1_{}_{}'.format(input_size,device))

    # plt.plot(layer,all,label='224x224')
    # a = [i[0] for i in DW]
    # b = [i[1] for i in DW] 
    # plt.scatter(a,b,marker="x")
    # a = [i[0] for i in conv]
    # b = [i[1] for i in conv] 
    # plt.scatter(a,b,marker="*")


    # plt.plot(layer,all,label='32x32')
    # a = [i[0] for i in DW]
    # b = [i[1] for i in DW] 
    # plt.scatter(a,b,marker="x")
    # a = [i[0] for i in conv]
    # b = [i[1] for i in conv] 
    # plt.scatter(a,b,marker="*")

    # net = MobileNet(n_class=10)
    # # ,kernel_list=[5, 7, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3]

    # model = list(net.modules())

    # idx = 1
    # for idx in range(len(model)):
    #     m = model[idx]
    #     m.old_forward = m.forward
    #     # print(type(new_forward(m)))
    #     m.forward = new_forward(m)

    # input = torch.randn(1,3,64,64)
    # net(input)

    # kernel_list = []
    # DW = []
    # conv = []
    # all = []
    # idx = 0
    # kernel_list = []
    # for i in range(len(model)):
    #     m = model[i]
    #     if type(m) == nn.Conv2d and m.groups == m.in_channels:
    #         DW.append([idx,m.flops/m.macs])
    #         all.append(m.flops/m.macs)
    #         print("DW",m.flops/m.macs)

    #         idx+=1
    #     elif type(m) == nn.Conv2d:
    #         conv.append([idx,m.flops/m.macs])
    #         all.append(m.flops/m.macs)
    #         print("Conv",m.flops/m.macs)
    #         idx+=1
    # layer = range(len(all))




    # print(kernel_list)

    # kernel_list = [3, 0, 3, 0, 3, 3, 0, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3]

    # net = MobileNetV2(n_class=25,kernel_list=kernel_list)
    # # stat(net,(3,224,224))
    # model = list(net.modules())

    # idx = 1
    # for idx in range(len(model)):
    #     m = model[idx]
    #     m.old_forward = m.forward
    #     # print(type(new_forward(m)))
    #     m.forward = new_forward(m)

    # input = torch.randn(1,3,224,224)
    # net(input)

    # kernel_list = []
    # for i in range(len(model)):
    #     m = model[i]
    #     if type(m) == nn.Conv2d and m.groups == m.in_channels:
    #         if m.flops/m.macs > 3:
    #             kernel_list.append(3)
    #         else:
    #             kernel_list.append(0)
    # # print(net)
    # # print(kernel_list)


    # # kernel_list = [3, 5, 3, 5, 3, 5, 3, 3, 3, 3, 3, 5, 3]
    # # print()
    # net = MobileNetV2()
    # print(stat(net,(3,32,32)))

