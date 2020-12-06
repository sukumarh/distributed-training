# -*- coding: utf-8 -*-
import json

class Configurations:
    def __init__(self, distributed_configs=True):
        self.distributed_configs = distributed_configs

        if not distributed_configs:
            self.config = {
                'config_index': 0,

                # model
                'model_name': '',

                # hardware
                'is_distributed': False,
                'distributed_strategy': 'None',
                'num_of_nodes': 1,
                'num_of_gpus': 1,

                # hyper-parameters
                'num_epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.1
            }
        
            self.model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101']
            self.batch_size = [32, 128, 512]

        else:
            # Distributed
            self.config = {
                'config_index': 0,

                # model
                'model_name': 'resnet18',

                # hardware
                'is_distributed': True,
                'distributed_strategy': 'nccl',
                'num_of_gpus': 1,

                # dataset
                'distribute_data': False,

                # hyper-parameters
                'num_epochs': 30,
                'batch_size': 128,
                'learning_rate': 0.1
            }
            self.model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101']
            self.batch_size = [128, 512]
            self.num_of_gpus = [2, 3, 4]
            self.distribute_data = [False, True]
        
        
    def generateConfigurations(self, existing_configs=None, filename='configurations.json'):
        # configurations = {'config': []}
        configurations = []
        empty = True
        if existing_configs is not None:
            configurations = existing_configs
            empty = False
        config = self.config.copy()

        if not self.distributed_configs:
            for model in self.model_names:
                config['model_name'] = model
                for batch in self.batch_size:
                    config['batch_size'] = batch
                    if empty:
                        empty = False
                    else:
                        config['config_index'] = configurations[-1]['config_index'] + 1
                    # configurations['config'].append(config.copy())
                    configurations.append(config.copy())
        else:
            for dist_data in self.distribute_data:
                config['distribute_data'] = dist_data
                for model in self.model_names:
                    config['model_name'] = model
                    for n_gpus in self.num_of_gpus:
                        config['num_of_gpus'] = n_gpus
                        for batch in self.batch_size:
                            config['batch_size'] = batch
                            if empty:
                                empty = False
                            else:
                                config['config_index'] = configurations[-1]['config_index'] + 1
                            # configurations['config'].append(config.copy())
                            configurations.append(config.copy())

        with open(filename, 'w') as file:
            json.dump(configurations, file)
            
    def loadConfigurations(self, filename='configurations.json'):
        with open(filename, 'r') as file:
            configurations = json.load(file)
        return configurations
            
if __name__ == '__main__':
    config = Configurations(distributed_configs=True)
    #config.generateConfigurations()
    configurations = config.loadConfigurations()
    config.generateConfigurations(existing_configs=configurations)