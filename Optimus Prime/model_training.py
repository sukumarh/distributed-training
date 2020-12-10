# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
import glob
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error
import pickle


files = glob.glob("../Data Collection/training_data/*.csv")
dataset = []
for file in tqdm(files):
    data = pd.read_csv(file)
    dataset.append(data)

dataset = pd.concat(dataset, axis=0, ignore_index=True)

# Select features
select_X_columns = ['num_of_paramters','gpu', 'is_distributed', 
                    'num_workers_data_loader', 'num_of_gpus', 'batch_size']
select_y_columns = ['epoch_timings']

X, y =  dataset[select_X_columns], dataset[select_y_columns]

#Preprocess GPU feature
device_list = ['P40', 'P100', 'V100']
for i, device_name in enumerate(X['gpu']):
    for device in device_list:
        if device in device_name:
            X['gpu'][i] = device
            break
gpu = pd.get_dummies(X['gpu'], drop_first=False)
X = pd.concat([X, gpu], axis=1)
X.drop(['gpu'], axis=1, inplace=True)

# Initializing the model
regressor = LGBMRegressor(boosting_type='gbdt', max_depth=- 1)

# Hyperparameter optimization 
kfold = KFold(n_splits=5, shuffle=True, random_state=42).split(X=X, y=y)
param_grid = {
    'num_leaves': [10, 31, 127],
    'reg_alpha': [0.1, 0.5],
    'learning_rate': [0.1, 0.5, 0.01],
    'n_estimators': [5, 10, 20]
}
gsearch = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=kfold)
lgb_model = gsearch.fit(X=X, y=y)

print(lgb_model.best_params_, lgb_model.best_score_)

# Evaluating the model
for param in lgb_model.best_params_:
    setattr(regressor, param, lgb_model.best_params_[param])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

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