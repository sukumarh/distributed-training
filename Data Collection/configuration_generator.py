# -*- coding: utf-8 -*-
import json

class Configurations:
    def __init__(self):
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
            'learning_rate': 0.01
        }        
        
        self.model_names = [ 'resnet18', 'resnet34', 'resnet50', 'resnet101']
        self.batch_size = [32, 128, 512]
        
    def generateConfigurations(self, filename='configurations.json'):
        # configurations = {'config': []}
        configurations = []
        config = self.config.copy()
        
        for model in self.model_names:
            config['model_name'] = model
            for batch in self.batch_size:
                config['batch_size'] = batch
                # configurations['config'].append(config.copy())
                configurations.append(config.copy())
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