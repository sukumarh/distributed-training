import os
import sys
import argparse
import pandas as pd
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.nn.parallel import DistributedDataParallel as DDP

from configuration_generator import Configurations

data_directory = 'data/'
save_directory = 'training_data/'
num_workers_data_loaders = 8


def str2bool(s):
    return s.lower() in ('true', '1')


def load_config(config=None):
    config_ = {
        'config_index': 0,

        # model
        'model_name': 'resnet18',
        'num_of_paramters': 0,
        'num_of_layers': 0,
        'model_type': 'cnn',

        # hardware
        'gpu': '',
        'is_distributed': True,
        'distributed_strategy': 'None',
        'num_of_nodes': 1,
        'num_of_gpus': 1,

        # dataset
        'dataset': 'cifar10',
        'num_workers_data_loader': 8,

        # hyper-parameters
        'num_epochs': 2,
        'batch_size': 512,
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'gamma': 0.1,
        'milestones': '60,80',
        'lr_scheduler': '',
        'optimizer': '',
        'criterion': ''
    }

    if config:
        for i in config:
            config_[i] = config[i]
    return config_


def load_cifar10_dataset():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    print('Loading train data')
    train_data = datasets.CIFAR10(root=data_directory,
                                  train=True,
                                  transform=train_transform,
                                  download=True)

    print('\nLoading test data')
    test_data = datasets.CIFAR10(root=data_directory,
                                 train=False,
                                 transform=test_transform,
                                 download=True)

    return train_data, test_data


def get_dataloaders(train_data, test_data, config):
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=num_workers_data_loaders)

    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=config['batch_size'],
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers_data_loaders)
    return train_loader, test_loader


# Distributed Data Parallel setup
def setup(rank, world_size,
          address='localhost',
          port='12335',
          backend='nccl',
          gloo_file="file:///{your local file path}"):
    if sys.platform == 'win32' and backend == 'gloo':
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        init_method = gloo_file

        # initialize the process group
        dist.init_process_group(
            backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size
        )
    else:
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port

        # initialize the process group
        dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def test(cnn, criterion, training_log, test_loader, rank):
    cnn.eval()
    correct, total = 0., 0.
    xentropy_loss_avg_val = 0.

    test_start_time = time()
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(rank)
        labels = labels.to(rank)

        with torch.no_grad():
            pred = cnn(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss_avg_val += xentropy_loss.item()

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    training_log['test_timings'].append(time() - test_start_time)
    training_log['test_error'].append(xentropy_loss_avg_val / (i + 1))
    val_acc = correct / total
    cnn.train()
    return val_acc


def train(rank, world_size, train_data, test_data, setup_config={}, config=None):
    """
    Train function

    Capture the training statistics
    rank            - GPU index
    world_size      - Total number of GPUs
    train_data      - Training data
    test_data       - Test data
    setup_config    - Configuration for setting up distributed environment
    config          - Training configuration
    """
    print(f'Setting up DDP for GPU (rank): {rank}.')
    setup(rank=rank, world_size=world_size, **setup_config)

    torch.cuda.set_device(rank)

    config = load_config(config)
    config['learning_rate'] = 0.1
    config['batch_size'] = 512

    train_loader, test_loader = get_dataloaders(train_data, test_data, config)

    training_log = {'train_acc': [],
                    'test_acc': [],
                    'train_loss': [],
                    'test_error': [],
                    'epoch_timings': [],
                    'test_timings': []
                    }

    model = models.__getattribute__(config['model_name'])()

    if 'fc' in model.__dir__():
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(train_loader.dataset.classes))
    elif 'classifier' in model.__dir__():
        last_layer = len(model.classifier) - 1
        num_ftrs = model.classifier[last_layer].in_features
        model.classifier[last_layer] = nn.Linear(num_ftrs, len(train_loader.dataset.classes))

    model_cuda = model.to(rank)

    cnn = DDP(model_cuda, device_ids=[rank], output_device=rank)

    config['num_of_paramters'] = sum(p.numel() for p in cnn.parameters())
    config['gpu'] = torch.cuda.get_device_name()

    num_epochs = config['num_epochs']

    criterion = nn.CrossEntropyLoss().to(rank)
    cnn_optimizer = optim.SGD(cnn.parameters(),
                              lr=config['learning_rate'],
                              momentum=config['momentum'],
                              nesterov=True,
                              weight_decay=config['weight_decay'])
    scheduler = lr_scheduler.MultiStepLR(cnn_optimizer,
                                         list(map(lambda x: int(x),
                                                  config['milestones'].split(','))),
                                         gamma=config['gamma'])

    p = len(train_loader) // 47
    for epoch in range(num_epochs):
        epoch_start_time = time()

        print(' ' * 150, end='\r')
        __print__, __result__ = f'[GPU: {rank}] Epoch {epoch + 1}/{num_epochs}: ', ''
        print(__print__, end='\r')

        xentropy_loss_avg = 0.
        correct = 0.
        total = 0.

        for i, (images, labels) in enumerate(train_loader):

            if i % p == 0:
                __print__ += '#'
                print(__print__ + __result__, end='\r')

            images = images.to(rank)
            labels = labels.to(rank)

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
                                                           accuracy * 100) + '%'
            print(__print__ + __result__, end='\r')

        test_acc = test(cnn, criterion, training_log, test_loader, rank)
        scheduler.step()

        training_log['epoch_timings'].append(time() - epoch_start_time)
        training_log['train_acc'].append(accuracy)
        training_log['test_acc'].append(test_acc)
        training_log['train_loss'].append(xentropy_loss_avg / (i + 1))
    print()

    save_data(training_log, config, rank)
    cleanup()


def save_data(training_log, config, rank):
    # Naming scheme: Model_GPU_Rank_{Distributed Strategy (NCCL)}_{config-index}.csv
    # Example: resnet18_V100_2_nccl_24.csv
    if config['is_distributed']:
        file_name = '{}_{}_{}_{}_{}.csv'.format(config['model_name'],
                                                config['gpu'],
                                                rank,
                                                config['distributed_strategy'],
                                                config['config_index'])
    else:
        file_name = '{}_{}_{}.csv'.format(config['model_name'],
                                          config['gpu'],
                                          config['config_index'])

    saving_data = pd.DataFrame.from_dict(training_log)
    for key in config:
        saving_data[key] = config[key]
    saving_data.to_csv(save_directory + file_name, index=True)
    saving_data.tail()
    print(f'[GPU: {rank}]Training data saved successfully.')


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser(description="Data Collection trainer (PyTorch)")

    parser.add_argument('-b', '--batch-size', type=int, default=512,
                        help="Batch Size")
    parser.add_argument('-c', '--configurations', type=str, default='',
                        help="Comma-separated list of configurations (config-index) to run. ")
    parser.add_argument('--configuration-file', type=str, default='',
                        help="The configuration file to refer to.")
    parser.add_argument('-d', '--data', type=str, default='data/',
                        help="The location of the dataset.")
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help="The dataset like CIFAR10.")
    parser.add_argument('-da', '--distributed-address', type=str, default='localhost',
                        help="Address for distributed process group setup. ")
    parser.add_argument('-dp', '--distributed-port', type=str, default='12335',
                        help="Port for distributed process group setup. ")
    parser.add_argument("--distributed-backend", default='nccl', type=str,
                        help='Distributed backend like NCCL, Gloo.')
    parser.add_argument('-e', '--epochs', default=2, type=int,
                        help='Number of epochs.')
    parser.add_argument('-g', "--gloo-file", default=None, type=str,
                        help='Gloo file location, if backend chosen is gloo.')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.1,
                        help="Learning Rate")
    parser.add_argument('-m', '--model-name', type=str, default='resnet18',
                        help="The model to be trained.")
    parser.add_argument("--num-nodes", default=1, type=int,
                        help='Number of nodes.')
    parser.add_argument("--num-gpus", default=1, type=int,
                        help='Number of GPUs.')
    parser.add_argument('-w', "--num-workers", default=16, type=int,
                        help='Number of workers for the data loaders.')
    parser.add_argument('-s', '--save-location', default='training_data/', type=str,
                        help='Save location of the training log.')

    args = parser.parse_args()

    data_directory = args.data
    save_directory = args.save_location
    num_workers_data_loaders = args.num_workers

    setup_config = {
        'address': args.distributed_address,
        'port': args.distributed_port,
        'backend': args.distributed_backend,
        'gloo_file': args.gloo_file
    }

    if args.configurations != '':

        cf = Configurations()
        if args.configuration_file == '':
            configs = cf.loadConfigurations()
        else:
            configs = cf.loadConfigurations(args.configuration_file)

        # Check for GPU support
        if torch.cuda.is_available():
            print('CUDA Toolkit available for PyTorch')
            print('GPU: ' + torch.cuda.get_device_name() + '\n')

            n_gpus = torch.cuda.device_count()
            world_size = configs['num_of_gpus']

            train_data, test_data = load_cifar10_dataset()

            for config_index in args.configurations.split(','):
                print('\n' + ('-' * 28) + '-' * len(str(config_index)) +
                      f'\n> Running configuration: {config_index}  <\n' +
                      ('-' * 28) + '-' * len(str(config_index)))

                config = configs[int(config_index)]
                if n_gpus < world_size:
                    print('{} GPUs required but only {} found. Skipping configuration {}.'.format(world_size,
                                                                                                  n_gpus,
                                                                                                  config_index))
                    continue

                print('\n' + ('#' * 40) +
                      f'\n# GPUs available: {n_gpus}, GPUs Required: {world_size} \n' +
                      ('#' * 40))

                print('\n#' + ('-' * 40) +
                      '\n# {} on {} \n'.format(config['model_name'],
                                               torch.cuda.get_device_name()) +
                      '#' + ('-' * 40) + '\n')

                config['num_workers_data_loader'] = args.num_workers

                mp.spawn(train,
                         args=(world_size, train_data, test_data, setup_config, config),
                         nprocs=world_size,
                         join=True)
                print()

        else:
            print('GPU Support not found for PyTorch')
            print('Exiting...')
    else:
        config = {
            'batch_size': args.batch_size,
            'dataset': args.dataset,
            'is_distributed': True,
            'distributed_strategy': args.distributed_backend,
            'num_epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'model_name': args.model_name,
            'num_of_nodes': args.num_nodes,
            'num_of_gpus': args.num_gpus,
            'num_workers_data_loader': args.num_workers
        }

        n_gpus = torch.cuda.device_count()
        world_size = args.num_gpus
        if n_gpus >= world_size:
            print('\n' + ('#' * 40) +
                  f'\n# GPUs available: {n_gpus}, GPUs Required: {world_size} \n' +
                  ('#' * 40))

            print('\n#' + ('-' * 40) +
                  '\n# {} on {} \n'.format(config['model_name'],
                                           torch.cuda.get_device_name()) +
                  '#' + ('-' * 40))

            train_data, test_data = load_cifar10_dataset()
            print()
            mp.spawn(train,
                     args=(world_size, train_data, test_data, setup_config, config),
                     nprocs=world_size,
                     join=True)
        else:
            print('{} GPUs required but only {} found. Exiting...'.format(world_size, n_gpus))
