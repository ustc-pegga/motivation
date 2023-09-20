import os
import sys
import torch
from test1 import generate_data_test1
from model.model import MobileNetV2

def get_result(envpath,execute,datasets,device,threads):
    print('adb shell "{} && {} {} {} {}"'.format(envpath,execute,datasets,device,threads))
    result = os.popen('adb shell "{} && {} {} {} {}"'.format(envpath,execute,datasets,device,threads))
    context = result.read()
    return context

envpath = "export LD_LIBRARY_PATH=/data/local/tmp/lib:${LD_LIBRARY_PATH}"
execute = "./data/local/tmp/mindspore_op"

root = "/data/local/tmp/op/"

datasets = "/data/local/tmp/op/kernel/kernel/dwconv_k/ms"
device = "0"
threads = "1"

datasets = "imagenet"
type = 'mbv2'
device = 'mi11_core7'
src_path = "../data/test6/{}/{}/ms".format(type,datasets)
dst_path = "/data/local/tmp/data/test6/{}/{}".format(type,datasets)
txt_path = "../data/test6/{}/{}/data/{}.txt".format(type,datasets,device)
def get_device(device):
    if "gpu" in device:
        return "1"
    else:
        return "0"
    
def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def start_test6(src_path,dst_path,txt_path,device):

    os.system("adb push {} {}".format(src_path,dst_path))

    result = get_result(envpath,execute,dst_path,get_device(device),threads)
    print(result)
    f = open(txt_path,'w')
    f.write(result)
    return result

result = start_test6(src_path,dst_path,txt_path,device)
print(result)