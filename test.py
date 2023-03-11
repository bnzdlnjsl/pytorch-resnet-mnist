import os
from torchinfo import summary
from os.path import join, exists
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from torchsummary import summary as summary_old
from model.resnet import *


def output_summary_old(resnet_size: int, output_path="summary\\"):
    if not exists(output_path):
        os.mkdir(output_path)

    print("summary pretrained...")
    resnet_name = "resnet" + str(resnet_size)
    model = torch.hub.load('pytorch/vision:v0.10.0', resnet_name, pretrained=True)
    model.to('cuda')
    model.eval()
    output_file = join(output_path, "summary_resnet-{}_pretrained.txt".format(resnet_size))
    summary_old(model, (3, 224, 224), output_file=output_file, verbose=False)
    with open(output_file, encoding='utf8') as f:
        l1 = f.readlines()

    print("summary myself...")
    device = torch.device('cuda')
    dtype = torch.float32
    model = Size2ResNet(resnet_size, 3, 1000, device=device, dtype=dtype)
    model.eval()
    output_file = join(output_path, "summary_resnet-{}_myself.txt".format(resnet_size))
    summary_old(model, (3, 224, 224), output_file=output_file, verbose=False)
    with open(output_file, encoding='utf8') as f:
        l2 = f.readlines()

    for i, (s1, s2) in enumerate(zip(l1, l2)):
        if s1 != s2:
            print("different in line [{}]:".format(i))
            print(s1)
            print(s2)
            break


def output_summary(resnet_size: int, output_path="summary\\"):
    if not exists(output_path):
        os.mkdir(output_path)

    print("summary pretrained...")
    resnet_name = "resnet" + str(resnet_size)
    model = torch.hub.load('pytorch/vision:v0.10.0', resnet_name, pretrained=True)
    model.to('cuda')
    model.eval()
    model_stats = summary(model, (1, 3, 224, 224), depth=10, verbose=False)

    output_file = join(output_path, "summary_resnet-{}_pretrained.txt".format(resnet_size))
    with open(output_file, 'w', encoding='utf8') as f:
        f.write(str(model_stats))

    print("summary myself...")
    device = torch.device('cuda')
    dtype = torch.float32
    model = Size2ResNet(resnet_size, 3, 1000, device=device, dtype=dtype)
    model.eval()
    model_stats = summary(model, (1, 3, 224, 224), depth=10, verbose=False)

    output_file = join(output_path, "summary_resnet-{}_myself.txt".format(resnet_size))
    with open(output_file, 'w', encoding='utf8') as f:
        f.write(str(model_stats))


def summary_all():
    size_list = [18, 34, 50, 101, 152]
    for size in size_list:
        output_summary(size)


# output_summary_old(152)
# output_summary(152)
summary_all()
