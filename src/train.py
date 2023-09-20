# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

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
from model.model import MobileNet
from tensorboardX import SummaryWriter
import json
from lib.utils import accuracy, AverageMeter, progress_bar, get_output_folder
from lib.data import get_dataset
from lib.net_measure import measure_model
from model.model import *
from pruning import *
from tran import tran
def parse_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--model', default='mobilenet', type=str, help='name of the model to train')
    parser.add_argument('--dataset', default='imagenet', type=str, help='name of the dataset to train')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of GPUs to use')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--n_worker', default=4, type=int, help='number of data loader worker')
    parser.add_argument('--lr_type', default='exp', type=str, help='lr scheduler (exp/cos/step3/fixed)')
    parser.add_argument('--n_epoch', default=150, type=int, help='number of epochs to train')
    parser.add_argument('--wd', default=4e-5, type=float, help='weight decay')
    parser.add_argument('--seed', default=None, type=int, help='random seed to set')
    parser.add_argument('--data_root', default=None, type=str, help='dataset path')
    # resume
    parser.add_argument('--ckpt_path', default=None, type=str, help='checkpoint path to resume from')
    # run eval
    parser.add_argument('--eval', action='store_true', help='Simply run eval')
    parser.add_argument('--kernel', default='3', type=int, help='kernel_size')
    return parser.parse_args()


def get_model(model_type,n_class,kernel_size):
    print('=> Building model..')
    if model_type == 'mbv1':
        net = MobileNet(n_class=n_class,kernel_size = kernel_size)
    else:
        net = MobileNetV2(n_class=n_class,kernel_size = kernel_size)
    return net.cuda() if use_cuda else net


def train(epoch, train_loader):
    print('\nEpoch: %d' % epoch)
    net.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # timing
        batch_time.update(time.time() - end)
        end = time.time()

        progress_bar(batch_idx, len(train_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
                     .format(losses.avg, top1.avg, top5.avg))
    writer.add_scalar('loss/train', losses.avg, epoch)
    writer.add_scalar('acc/train_top1', top1.avg, epoch)
    writer.add_scalar('acc/train_top5', top5.avg, epoch)


def test(epoch, test_loader, save=True):
    global best_acc
    net.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # timing
            batch_time.update(time.time() - end)
            end = time.time()

            progress_bar(batch_idx, len(test_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
                         .format(losses.avg, top1.avg, top5.avg))

    if save:
        writer.add_scalar('loss/test', losses.avg, epoch)
        writer.add_scalar('acc/test_top1', top1.avg, epoch)
        writer.add_scalar('acc/test_top5', top5.avg, epoch)

        is_best = False
        if top1.avg > best_acc:
            best_acc = top1.avg
            is_best = True

        print('Current best acc: {}'.format(best_acc))
        save_checkpoint({
            'epoch': epoch,
            'model': args.model,
            'dataset': args.dataset,
            'state_dict': net.module.state_dict() if isinstance(net, nn.DataParallel) else net.state_dict(),
            'acc': top1.avg,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint_dir=log_dir)
    return epoch, best_acc,loss


def adjust_learning_rate(optimizer, epoch):
    lr_type = "cos"
    if lr_type == 'cos':  # cos without warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / n_epoch))
    elif lr_type == 'exp':
        step = 1
        decay = 0.96
        lr = lr * (decay ** (epoch // step))
    elif lr_type == 'fixed':
        lr = lr
    else:
        raise NotImplementedError
    print('=> lr: {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(state, is_best, checkpoint_dir='.'):
    filename = os.path.join(checkpoint_dir, 'ckpt.pth.tar')
    print('=> Saving checkpoint to {}'.format(filename))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pth.tar', '.best.pth.tar'))


if __name__ == '__main__':
    args = parse_args()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        gc.collect()
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    seed = 2024
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = "mbv2"
    dataset = "cifar100"
    batch_size = 64
    n_epoch = 200
    n_workers = 12
    kernel_size = 3# args.kernel
    # data_root = "/datasets/tiny-imagenet-200"
    data_root = "/datasets/cifar100"
    print('=> Preparing data..')
    train_loader, val_loader, n_class = get_dataset(dataset, batch_size, n_workers,
                                                    data_root)
    print(len(train_loader))


    # net = MobileNet(n_class=25,kernel_size=kernel_size)
    # # rate = [1.0, 0.75, 0.75, 0.6875, 0.75, 0.71875, 0.71875, 0.703125, 0.703125, 0.6875, 0.65625, 0.640625, 0.765625, 0.7890625, 0.5]
    # # net = mymodel.MobileNet(10,[3, 24, 52, 96, 96, 184, 180, 388, 384, 364, 348, 332, 340, 732, 304])
    # #rate = [1.0, 0.75, 0.8125, 0.78125, 0.75, 0.75, 0.765625, 0.7421875, 0.7265625, 0.7109375, 0.703125, 0.6796875, 0.671875, 0.4609375, 0.203125]
    # rate = [1.0, 0.75, 0.8125, 0.78125, 0.8125, 0.796875, 0.8125, 0.7578125, 0.7421875, 0.703125, 0.671875, 0.6328125, 0.640625, 0.3671875, 0.203125]

    # IMAGE_SIZE = 224
    # print(IMAGE_SIZE)
    # n_flops, n_params, n_macs = measure_model(net.cuda(), IMAGE_SIZE, IMAGE_SIZE)
    # print('=> Model Parameter: {:.3f} M, FLOPs: {:.3f}M, MACs: {:.3f}'.format(n_params / 1e6, n_flops / 1e6, n_macs / 1e6))

    # del net

    ckpt_path = None
    kernel_list = [3, 5, 3, 5, 3, 3, 5, 3, 3, 3, 3, 3, 3, 5, 3, 3, 3]
    net = MobileNetV2(n_class=100)
    # ckpt_path = "/home/hujie/code/motivation/checkpoint/mbv2_k3_imagenet25_300-run3/ckpt.best.pth.tar"
    # ckpt_path = "/home/hujie/code/motivation/checkpoint/mobilenet_k3_imagenet25_300-run7/ckpt.best.pth.tar"

    if ckpt_path is not None:  # assigned checkpoint path to resume from
        print('=> Resuming from checkpoint..')
        checkpoint = torch.load(ckpt_path)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_dict = {k: v for k, v in sd.items() if net.state_dict()[k].numel() == v.numel()}
        # sd.update(new_dict)
        missing_keys, unexpected_keys = net.load_state_dict(new_dict, strict=False)
        # for param in net.features.parameters():
        #     param.requires_grad = False
    # rate = [1.0, 0.625, 0.75, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.6944444444444444, 0.75, 0.7083333333333334, 0.75, 0.6666666666666666, 0.75, 0.6875, 0.875, 0.7083333333333334, 0.875, 0.5833333333333334, 0.875, 0.7395833333333334, 0.875, 0.6770833333333334, 0.875, 0.6319444444444444, 0.875, 0.5833333333333334, 0.875, 0.7083333333333334, 0.85, 0.6291666666666667, 0.75, 0.625, 0.875, 0.7208333333333333, 0.4375, 0.2]
    # net = mbv2_pruning(net,rate,input=(224,224))
    # net = tran()
    # net2 = MobileNetV2(n_class=25,kernel_list=kernel_list)
    # net = tran(net,net2)
    print(net)
    net.cuda()
    wd = 4e-5
    lr = 0.1
    criterion = nn.CrossEntropyLoss()
    print('Using SGD...')
    print('weight decay  = {}'.format(args.wd))
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    eval = 0

    if eval:  # just run eval
        print('=> Start evaluation...')
        test(0, val_loader, save=False)
    else:  # train
        print('=> Start training...')
        print('Training {} on {}...'.format(model, dataset))
        log_dir = get_output_folder('../checkpoint', '{}_{}_{}_kernel'.format(model, kernel_size, dataset,n_epoch))
        print('=> Saving logs to {}'.format(log_dir))
        # tf writer
        writer = SummaryWriter(logdir=log_dir)
        acc = dict()
        f = open('{}/acc.txt'.format(log_dir), 'w')
        for epoch in range(start_epoch, start_epoch + n_epoch):
            lr = adjust_learning_rate(optimizer, epoch)
            train(epoch, train_loader)
            epoch, best_acc, loss = test(epoch, val_loader)
            acc[epoch] = best_acc
            print(epoch, best_acc)
            f.write('epoch: {}, best_acc: {}, loss: {}\n'.format(epoch, best_acc,loss))

        

        writer.close()
        # print('=> Model Parameter: {:.3f} M, FLOPs: {:.3f}M, best top-1 acc: {}%'.format(n_params / 1e6, n_flops / 1e6, best_acc))

