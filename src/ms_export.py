import io
import sys
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.onnx
import torch.nn.init as init
import numpy as np
import onnxruntime
import os

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def onnx_export(net, input, path, name):
    onnx_path = "{}/onnx/".format(path)
    ms_path = "{}/ms/".format(path)
    mkdir(onnx_path)
    mkdir(ms_path)
    print(ms_path)
    # input = torch.tensor(input)
    torch.onnx.export(net,               # model being run
                    input,                         # model input (or a tuple for multiple inputs)
                    onnx_path + name+".onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'] # the model's output names
                    #   dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                    #                 'output' : {0 : 'batch_size'}})
                    )
    print(onnx_path)
    a = os.system("./converter_lite --fmk=ONNX --modelFile={} --outputFile={}".format(onnx_path + name+".onnx", ms_path + name))