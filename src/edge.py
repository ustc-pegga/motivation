import os
import sys
import torch
from test1 import generate_data_test1
from model.model import MobileNetV2
import argparse

def get_result(envpath,execute,datasets,device,threads):
    print('adb shell "{} && {} {} {} {}"'.format(envpath,execute,datasets,device,threads))
    result = os.popen('adb shell "{} && {} {} {} {}"'.format(envpath,execute,datasets,device,threads))
    context = result.read()
    return context

envpath = "export LD_LIBRARY_PATH=/data/local/tmp/lib:${LD_LIBRARY_PATH}"
execute = "./data/local/tmp/mindspore_op"

root = "/data/local/tmp/op/"

datasets = "/data/local/tmp/op/kernel/kernel/dwconv_k/ms"
threads = "1"

datasets = "imagenet"
type = 'mbv2'
device = 'mi11_core7'
# src_path = "../data/test6/{}/{}/ms".format(type,datasets)
# dst_path = "/data/local/tmp/data/test6/{}/{}".format(type,datasets)
# txt_path = "../data/test6/{}/{}/data/{}.txt".format(type,datasets,device)
def get_device(device):
    if "gpu" in device or "GPU" in device:
        return "1"
    elif "cpu" in device or "core" in device:
        return "0"
    else:
        return "2"
    
def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def start(src_path,dst_path,txt_path,device):

    os.system("adb shell rm -rf {} ".format(dst_path))
    os.system("adb push {} {}".format(src_path,dst_path))

    result = get_result(envpath,execute,dst_path,get_device(device),threads)
    print(result)
    f = open(txt_path,'w')
    f.write(result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--device', default='none', type=str, help='name of the dataset to train')
    parser.add_argument('--type', default=None, type=str, help='pruning')
    parser.add_argument('--data_root', default=None, type=str, help='root')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = args.device
    type = args.type
    data_root = args.data_root
    src_path = "{}/{}/ms".format(data_root,type)
    dst_path = "/data/local/tmp/data/{}".format(type)
    make_dirs("{}/{}/data/".format(data_root,type))
    txt_path = "{}/{}/data/{}.txt".format(data_root,type,device)
    result = start(src_path,dst_path,txt_path,device)
    print(result)