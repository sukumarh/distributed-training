# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
import glob
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error
import pickle


files = glob.glob("../Data Collection/training_data/*.csv")
dataset = []
for file in tqdm(files):
    data = pd.read_csv(file)
    dataset.append(data)

dataset = pd.concat(dataset, axis=0, ignore_index=True)

select_X_columns = ['num_of_paramters','gpu', 'is_distributed', 
                    'num_workers_data_loader', 'num_of_gpus', 'batch_size']
select_y_columns = ['epoch_timings']

X, y =  dataset[select_X_columns], dataset[select_y_columns]
gpu = pd.get_dummies(X['gpu'], drop_first=False)
X = pd.concat([X, gpu], axis=1)
X.drop(['gpu'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

regressor = LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=- 1,
                          learning_rate=0.1, n_estimators=10)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

MSE = mean_squared_error(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
RMSE = MSE**(1/2)
print("Validation results: MAE:%3f RMSE:%3f"%(MAE, RMSE))

# Train on entire dataset to save model
regressor.fit(X, y)
filename = 'optimus_prime_model.pkl'
pickle.dump(regressor, open(filename, 'wb'))