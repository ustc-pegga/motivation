import os
import sys
import torch
import argparse
import re

def get_result(cmd):
    result = os.popen(cmd)
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



def start(src_path,dst_path,txt_path,device,cmd):

    os.system("adb shell rm -rf {} ".format(dst_path))
    os.system("adb push {} {}".format(src_path,dst_path))

    result = get_result(cmd)
    f = open(txt_path,'w')
    get_time(result,f)
    f.write(result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--device', default='none', type=str, help='name of the dataset to train')
    parser.add_argument('--type', default=None, type=str, help='pruning')
    parser.add_argument('--data_root', default=None, type=str, help='root')
    parser.add_argument('--timeProfiling', default='false', type=str, help='profiling')
    return parser.parse_args()

def get_cmd(args,model):
    cmd = "adb shell ' {} && ./data/local/tmp/benchmark  \
        --modelFile=/data/local/tmp/data/{}/{} \
        --device={} \
        --timeProfiling={} \
        --numThreads=1 \
        --loopCount=1000 \
        --warmUpLoopCount=20 '".format(envpath,args.type,model,args.device,'true')
    
# format(args.model,args.device,args.threads,args.loopCount,args.warmUpLoopCount,args.inputShape)
    return cmd

def get_time(text,f):
    op_pattern = r'(\w+)\s+(\d+\.\d+)\s+\d+\.\d+\s+\d+\s+\d+\.\d+'

    # 在文本中查找算子信息
    matches = re.findall(op_pattern, text)

    # 构建算子信息字典
    operator_info = {}
    for match in matches:
        op_name = match[0]
        avg_time = match[1]
        operator_info[op_name] = avg_time

    print("Operator Information:")
    for op_name, avg_time in operator_info.items():
        f.write(f"{op_name}: {avg_time} ms\n")

    # 正则表达式模式，用于匹配最小、最大和平均运行时间
    time_pattern = r'MinRunTime = (\d+\.\d+) ms, MaxRuntime = (\d+\.\d+) ms, AvgRunTime = (\d+\.\d+) ms'

    # 在文本中查找时间信息
    match = re.search(time_pattern, text)

    if match:
        min_runtime = match.group(1)
        max_runtime = match.group(2)
        avg_runtime = match.group(3)
        f.write(f"Min Run Time: {min_runtime} ms\n")
        print("Min Run Time:", min_runtime, "ms")
        print("Max Run Time:", max_runtime, "ms")
        print("Avg Run Time:", avg_runtime, "ms")
    else:
        print("Time information not found in the text.")

if __name__ == "__main__":
    args = parse_args()
    cmd = get_cmd(args)
    device = args.device
    type = args.type
    data_root = args.data_root
    src_path = "{}/{}/ms".format(data_root,type)
    dst_path = "/data/local/tmp/data/{}".format(type)
    make_dirs("{}/{}/data/".format(data_root,type))
    txt_path = "{}/{}/data/{}.txt".format(data_root,type,device)
    result = start(src_path,dst_path,txt_path,device,cmd)
    print(result)