# -*- coding: utf-8 -*-
from configuration_generator import Configurations
import os

config = Configurations()
configurations = config.loadConfigurations()

for configuration in configurations['config']:
    print("Training with configuration: ",  configuration['config_index'])
    string_conf = str(configuration)
    os.system('python trainer_pytorch.py ' + string_conf)
    print('Completed')
