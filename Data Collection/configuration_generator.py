# -*- coding: utf-8 -*-
import json

class Configurations:
    def __init__(self):
        self.config = {
            'config_index': 0,
            
            # model
            'model_name': '',
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
            'dataset': '',
            
            # hyper-parameters
            'batch_size': 32,
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'gamma': 0.1,
            'milestones': '60,80',
            'lr_scheduler': '',
            'optimizer':'',
            'criterion': ''   
        }
        
        self.model_names = [ 'Alexnet', 'Resnet18', 'Resnet32', 'Resnet44/40', 'Resnet56']
        self.batch_size = [32, 128, 512]
        
    def generateConfigurations(self, filename='configurations.json'):
        configurations = {'config': []}
        config = self.config.copy()
        
        for model in self.model_names:
            config['model_name'] = model
            for batch in self.batch_size:
                config['batch_size'] = batch
                configurations['config'].append(config.copy())
                config['config_index'] += 1
                
        with open(filename, 'w') as file:
            json.dump(configurations, file)
            
    def loadConfigurations(self, filename='configurations.json'):
        with open(filename, 'r') as file:
            configurations = json.load(file)
        return configurations
            
if __name__ == '__main__':
    config = Configurations()
    config.generateConfigurations()
    configurations = config.loadConfigurations()