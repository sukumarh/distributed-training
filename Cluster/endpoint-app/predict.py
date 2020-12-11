# -*- coding: utf-8 -*-
import pandas as pd
import pickle

epoch_time_model_path = "optimus_prime_epoch_time.pkl"
epoch_time_model = pickle.load(open(epoch_time_model_path, 'rb'))

accuracy_model_path = "optimus_prime_epoch_accuracy.pkl"
accuracy_model = pickle.load(open(accuracy_model_path, 'rb'))


data = {'num_of_paramters': 11689500, 'is_distributed': False, 'num_workers_data_loader': 16,
       'num_of_gpus': 4, 'batch_size': 128, 'P40': 0, 'P100': 1, 'V100': 0}

test = {'num_of_paramters': '11689500', 'is_distributed': False, 'num_workers_data_loader': '4',
        'num_of_gpus': '1', 'batch_size': '128', 'P40': 0, 'P100': 1, 'V100': 0}

def predict_epoch_time(data=data):
    data = pd.DataFrame([data])
    epoch_time = epoch_time_model.predict(data)
    return epoch_time[0]

def predict_accuracy(data=data, max_epochs=100):
    for epoch in range(max_epochs):
        config = data.copy()
        config = pd.DataFrame([config])
        config.insert(0, 'epoch', epoch)
        config['epoch'] = epoch
        accuracy = accuracy_model.predict(config)
        if accuracy[0] >= 0.6:
            return epoch
    return -1

def predict(data=test):
    epoch_time = predict_epoch_time(data)
    epoch_no = predict_accuracy(data)
    return epoch_time, epoch_no