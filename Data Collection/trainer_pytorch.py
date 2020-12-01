import os
import pickle
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.autograd.profiler as profiler

from torchvision import datasets, models, transforms


def load_config(config=None):
    config_ = {
        'config_index': 0,

        # model
        'model_name': 'Resnet18',
        'num_of_paramters': 0,
        'num_of_layers': 0,
        'model_type': 'cnn',

        # hardware
        'gpu': '',
        'is_distributed': False,
        'distributed_strategy': 'None',
        'num_of_nodes': 1,
        'num_of_gpus': 1,

        # dataset
        'dataset': 'cifar10',
        'num_workers_data_loader': 0,

        # hyper-parameters
        'batch_size': 128,
        'learning_rate': 0.01,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'gamma': 0.1,
        'milestones': '60,80',
        'lr_scheduler': '',
        'optimizer':'',
        'criterion': ''   
    }
    
    if config:
        for i in config:
            config_[i] = config[i]
    return config_


def load_cifar10_dataset(config):
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([ 
        transforms.ToTensor(), 
        normalize 
    ])
    
    print('Loading train data')
    train_data = datasets.CIFAR10(root='data/',
                                  train=True,
                                  transform=train_transform,
                                  download=True)
    
    print('\nLoading test data')
    test_data = datasets.CIFAR10(root='data/',
                                  train=False,
                                  transform=test_transform,
                                  download=True)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=config['num_workers_data_loader'])

    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=config['batch_size'],
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=config['num_workers_data_loader'])
    return train_loader, test_loader


def run_training(config=None):
    ## Check for GPU support
    if torch.cuda.is_available():
        print('CUDA Toolkit available for PyTorch')
        print('GPU: ' + torch.cuda.get_device_name() + '\n')
    else:
        print('GPU Support not found for PyTorch')
        print('Exiting...')
        return
        
    config = load_config(config)
    train_loader, test_loader = load_cifar10_dataset(config)
    
    model = models.resnet18()
    
    config['num_of_paramters'] = sum(p.numel() for p in model.parameters())
    config['gpu'] = torch.cuda.get_device_name()
    
    trainer = torch_trainer(model, config, train_loader, test_loader)
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        trainer.train(1)
    print(prof)
    
    trainer.save_data()


class torch_trainer:
    
    def __init__(self, model, config, train_loader, test_loader):
        if torch.cuda.is_available():
            self.training_log = {'train_acc': [],
                                 'test_acc': [],
                                 'train_loss': [],
                                 'test_error': [],
                                 'epoch_timings': [],
                                 'test_timings': []
                                }
            self.config = config
            self.train_loader = train_loader
            self.test_loader = test_loader
            self.model = model
        else:
            print('GPU not found. Trainer not initialized.')
    
    def train(self, num_epochs):
        """
        Train function

        Capture the training statistics
        num_epochs - No. of epochs
        """        
        print('\n' + ('#' * 40) + 
              '\n# {} on GPU: {} \n'.format(self.config['model_name'], 
                                            self.config['gpu']) + 
              ('#' * 40))
        if 'fc' in self.model.__dir__():
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 
                                      len(self.train_loader.dataset.classes))
        
        cnn = self.model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        cnn_optimizer = optim.SGD(cnn.parameters(),
                                  lr=self.config['learning_rate'],
                                  momentum=self.config['momentum'], 
                                  nesterov=True,
                                  weight_decay=self.config['weight_decay'])
        scheduler = lr_scheduler.MultiStepLR(cnn_optimizer, 
                                             list(map(lambda x: int(x), 
                                                      self.config['milestones'].split(','))),
                                             gamma=self.config['gamma'])

        print_bound, print_output = 3, True
        p = len(self.train_loader) // 60

        training_start_time = time()
        for epoch in range(num_epochs):
            epoch_start_time = time()

            if epoch < print_bound or epoch >= num_epochs - print_bound:
                print_output = True
            elif epoch == print_bound:
                print('.\n.\n.\n')
                print_output = False

            __print__, __result__ = '', ''
            print(' ' * 150, end='\r')
            if print_output:
                print(f'Epoch {epoch + 1}/{num_epochs}')
                print('-' * 10)
            else:
                __print__ = f'Epoch {epoch + 1}/{num_epochs}: '
                print(__print__, end='\r')

            xentropy_loss_avg = 0.
            correct = 0.
            total = 0.

            for i, (images, labels) in enumerate(self.train_loader):

                if i % p == 0:
                    __print__ += '#'
                    print(__print__ + __result__, end='\r')

                images = images.cuda()
                labels = labels.cuda()

                cnn.zero_grad()
                pred = cnn(images)

                xentropy_loss = criterion(pred, labels)
                xentropy_loss.backward()
                cnn_optimizer.step()

                xentropy_loss_avg += xentropy_loss.item()

                # Calculate running average of accuracy
                pred = torch.max(pred.data, 1)[1]
                total += labels.size(0)
                correct += (pred == labels.data).sum().item()
                accuracy = correct / total

                __result__ = '    Loss=%.3f, Accuracy=%.3f' % (xentropy_loss_avg / (i + 1), 
                                                               accuracy*100) + '%'
                print(__print__ + __result__, end='\r')


            test_acc = self._test(cnn, criterion)
            if print_output:
                print('\n')

            scheduler.step()
            
            self.training_log['epoch_timings'].append(time() - epoch_start_time)
            self.training_log['train_acc'].append(accuracy)
            self.training_log['test_acc'].append(test_acc)
            self.training_log['train_loss'].append(xentropy_loss_avg / (i + 1))
            
    def _test(self, cnn, criterion):        
        cnn.eval()
        correct, total = 0., 0.
        xentropy_loss_avg_val = 0.
        
        test_start_time = time()
        for i, (images, labels) in enumerate(self.test_loader):
            images = images.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                pred = cnn(images)

            xentropy_loss = criterion(pred, labels)
            xentropy_loss_avg_val += xentropy_loss.item() 

            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        self.training_log['test_timings'].append(time() - test_start_time)
        self.training_log['test_error'].append(xentropy_loss_avg_val / (i + 1))
        val_acc = correct / total
        cnn.train()
        return val_acc
    
    def visualize_training(self):
        pass
    
    def save_data(self):
        # Naming scheme: Model_GPU_{Distributed Strategy (NCCL)}_{config-index}.csv
        if self.config['is_distributed']:
            file_name = '{}_{}_{}_{}.csv'.format(self.config['model_name'], 
                                                 self.config['gpu'],
                                                 self.config['distributed_strategy'], 
                                                 self.config['config_index'])
        else:
            file_name = '{}_{}_{}.csv'.format(self.config['model_name'], 
                                              self.config['gpu'], 
                                              self.config['config_index'])
            
        saving_data = pd.DataFrame.from_dict(self.training_log)
        for key in self.config:
            saving_data[key] = self.config[key]
        saving_data.to_csv(file_name, index = True)
        saving_data.tail()
        
    def load_data(self):
        pass
    

if __name__ == "__main__":
    # args
    # num_workers_data_loader
    # configs
    
    run_training()