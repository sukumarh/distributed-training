# -*- coding: utf-8 -*-
import pandas as pd
import pickle

model_path = "optimus_prime_model.pkl"
model = pickle.load(open(model_path, 'rb'))

data = {'num_of_paramters': 11689500, 'is_distributed': False, 'num_workers_data_loader': 4,
       'num_of_gpus': 1, 'batch_size': 128, 'P40': 0, 'P100': 1, 'V100': 0}

def predict(data=data):
    data = pd.DataFrame([data])
    epoch_time = model.predict(data)
    return epoch_time

time = predict()