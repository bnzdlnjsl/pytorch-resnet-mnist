import os
from torchinfo import summary
from os.path import join, exists
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary as summary_old
from model.resnet import *
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import sampler


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

def MNIST_test():
    ROOT = 'data'
    batch_size = 128
    train_dataset = datasets.MNIST(ROOT, train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    for idx, (x, y) in enumerate(train_loader):
        print(idx, type(x), x.size(), x.dtype)
        print(idx, type(y), y.size(), y.dtype)
        break
        
        # if idx > (60000 // 128 + 3):
        #     raise StopIteration("non-stop iteration on mnist")  # never reached, it does stop
    
# MNIST_test()
    
def CIFAR10_test():
    NUM_TRAIN = 45000
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    ROOT = 'data'
    cifar10_train = datasets.CIFAR10(ROOT, train=True, download=True, transform=transform)
    loader_train = DataLoader(cifar10_train, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

    cifar10_val = datasets.CIFAR10(ROOT, train=True, download=True, transform=transform)
    loader_val = DataLoader(cifar10_val, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

    cifar10_test = datasets.CIFAR10(ROOT, train=False, download=True, transform=transform)
    loader_test = DataLoader(cifar10_test, batch_size=64)

# CIFAR10_test()

def calculate_mnist_mean_and_std():
    ROOT = 'data'
    train_dataset = datasets.MNIST(ROOT, train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128)
    test_dataset = datasets.MNIST(ROOT, train=False, transform=transforms.ToTensor(), download=True)
    test_loader = DataLoader(dataset=train_dataset, batch_size=128)
    mean = 0
    for _, (x, _) in enumerate(train_loader):
        x = x.numpy()
        mean += np.sum(x)
    for _, (x, _) in enumerate(test_loader):
        x = x.numpy()
        mean += np.sum(x)
    mean /= 70000 * 28 * 28
    std = 0
    for _, (x, _) in enumerate(train_loader):
        x = x.numpy()
        std += np.sum(np.square(x - mean))
    for _, (x, _) in enumerate(test_loader):
        x = x.numpy()
        std += np.sum(np.square(x - mean))
    std /= 70000 * 28 * 28
    std = np.sqrt(std)
    print("mean: {:.16f}".format(mean)) # mean: 0.2239893917339536
    print("std: {:.16f}".format(std))   # std: 0.4215089302415015

# calculate_mnist_mean_and_std()

def test_tensor_size_and_shape_type():
    t = torch.zeros(8, 16)
    print(type(t.size()))
    print(type(t.shape))
    print(t.shape == t.size())

# test_tensor_size_and_shape_type()