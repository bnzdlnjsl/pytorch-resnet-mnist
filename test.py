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

# output_summary_old(152)

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

# output_summary(152)

def summary_all():
    size_list = [18, 34, 50, 101, 152]
    for size in size_list:
        output_summary(size)

# summary_all()

def understand_module_forward_implementaion():
    class TrashModule(nn.Module):
        def __init__(self):
            super(TrashModule, self).__init__()
            return
        def this_is_another_forward(x):
            pass
        
    # print(type(TrashModule().forward))
    # print(TrashModule().forward.__name__ == nn.modules.module._forward_unimplemented.__name__)
    print(TrashModule().forward.__name__)

# understand_module_forward_implementaion()

def the_keyword_yield():
    def my_generator():
        for i in range(10):
            yield i * i
    
    # print(type(my_generator()))
    gen = my_generator()
    # print(type(gen))
    for i in gen:
        print(i)

# the_keyword_yield()

def method_parameters_and_instance_variable_weight():
    
    some_module = nn.Linear(128, 256)
    
    print("- weight:")
    print("\t", type(some_module.weight), some_module.weight.size())
    
    print("- all parameters:")
    for param in some_module.parameters():
        print("\t", type(param), param.size())
        
    print("- all parameters with their names:")
    for name, param in some_module.named_parameters():
        print("\t", type(param), param.size(), name)

# method_parameters_and_instance_variable_weight()