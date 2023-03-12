import os
import torch.nn as nn
import torch.nn.functional as F
import numpy
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import sampler

def get_mnist_loader(mode: str):
    ROOT = 'data'
    NUM_TRAIN = 55000
    BATCH_SIZE = 128
    TRANSFORM = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2239893917339536, ), (0.4215089302415015, ))
            ])
    
    assert NUM_TRAIN < 60000, "only 60000 training data, got {}".format(NUM_TRAIN)
    
    if mode == 'train':
        train_data = datasets.MNIST(ROOT, train=True, transform=TRANSFORM, download=True)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                                  sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
        
        return train_loader
    
    elif mode == 'val':
        val_data = datasets.MNIST(ROOT, train=True, transform=TRANSFORM, download=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE,
                                sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 60000)))
        
        return val_loader
    
    elif mode == 'test':
        test_data = datasets.MNIST(ROOT, train=False, transform=TRANSFORM, download=True)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
        
        return test_loader
    
    else:
        
        raise ValueError("invalid evaluation mode")

def get_cifar10_loader(mode: str):
    ROOT = 'data'
    NUM_TRAIN = 45000
    BATCH_SIZE = 64
    TRANSFORM = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    
    assert NUM_TRAIN < 50000, "only 50000 training data, got {}".format(NUM_TRAIN)
    
    if mode == 'train':
        cifar10_train = datasets.CIFAR10(ROOT, train=True, download=True, transform=TRANSFORM)
        loader_train = DataLoader(cifar10_train, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
        
        return loader_train
    
    elif mode == 'val':
        cifar10_val = datasets.CIFAR10(ROOT, train=True, download=True, transform=TRANSFORM)
        loader_val = DataLoader(cifar10_val, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
        
        return loader_val
    
    elif mode == 'test':
        cifar10_test = datasets.CIFAR10(ROOT, train=False, download=True, transform=TRANSFORM)
        loader_test = DataLoader(cifar10_test, batch_size=BATCH_SIZE)
        
        return loader_test
    
    else:
        
        raise ValueError("invalid evaluation mode")

def get_loader(dataset_name: str, mode: str):
    if dataset_name not in ["mnist", "cifar10"] or mode not in ["train", "val", "test"]:
        raise ValueError("invalid dataset name or evaluation mode")
    
    if dataset_name == "mnist":
        return get_mnist_loader(mode)
    elif dataset_name == "cifar10":
        return get_cifar10_loader(mode)