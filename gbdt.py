#%%
import pandas as pd
import os
os.chdir('C:/Users/33781/Downloads/M2 FTD S2/Applied Machine Learning')
import numpy as np
from data_processing import *
#%%

df = pd.read_csv('credit_risk_dataset.csv')
data = data_processing(df)
data()
df = data.df
#%%

import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from metrics import *

y = pd.DataFrame(df['loan_status'])
X = df.drop(y, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
train = lgb.Dataset(X_train, label = y_train)

params = {'objective':'binary',
          'seed': 42,
          'metric':'binary_logloss',
          'verbosity':-1
          }
#%%

#Training a baseline GBDT
model = lgb.train(params, train, 1000)
y_pred = np.round(model.predict(X_test))

print('Precision Baseline',precision_score(y_test, y_pred, pos_label=0))
print('Accuracy Baseline',accuracy_score(y_test, y_pred))
print('Recall Baseline',recall_score(y_test, y_pred, pos_label=0))

#%%

#Cross-validation to see how-well the model generalise
kf = KFold(n_splits=10)
precision_list = []
accuracy_list = []
recall_list = []
for i, (train_index, test_index) in enumerate(kf.split(X)):    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    train = lgb.Dataset(X_train, label = y_train) 
    model = lgb.train(params, train, 1000)

    y_pred = np.round(model.predict(X_test))

    precision_list.append(precision_score(y_test, y_pred, pos_label=0))
    accuracy_list.append(accuracy_score(y_test, y_pred))
    recall_list.append(recall_score(y_test, y_pred, pos_label=0))

print('Precision Baseline CV',np.mean(precision_list))
print('Accuracy Baseline CV',np.mean(accuracy_list))
print('Recall Baseline CV',np.mean(recall_list))
#%%
#Hyperparameters fine-tuning to find the best set of hyperparameters
#with the use of a validation set to prevent overfitting

import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score
from metrics import *
from datetime import datetime

y = pd.DataFrame(df['loan_status'])
X = df.drop(y, axis=1)


def objective(trial):
    print(datetime.now())

    lr = trial.suggest_float('learning_rate', 0.001, 1)
    max_depth = trial.suggest_int('max_depth', 20, 40)
    num_leaves = trial.suggest_int('num_leaves', 31, 400)
    lambda_1 = trial.suggest_float('lambda_1', 0, 1)
    lambda_2 = trial.suggest_float('lambda_2', 0, 1)
    max_bins = trial.suggest_int('max_bins', 255, 1024)
    
    params={'objective':'binary',
            'seed':42,
            'metric':'None',
            'device':'cpu',
            'learning_rate':lr,
            'max_depth':max_depth,
            'num_leaves':num_leaves,
            'max_bins':max_bins,
            'lambda_l1':lambda_1,
            'lambda_l2':lambda_2,
            'verbosity':-1}
    
    kfold = KFold(n_splits=5)
    precision_list = []
    f1_list = []
    for train_index, test_index in kfold.split(X):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_temp, y_temp = X.iloc[test_index], y.iloc[test_index]
        X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        del X_temp, y_temp
        
        train = lgb.Dataset(X_train, label=y_train)
        valid = lgb.Dataset(X_valid, label=y_valid, reference=train)
        
        model = lgb.train(params, train, 3000, valid_sets=[valid], feval=precision_metric, callbacks=[lgb.early_stopping(50)])
        
        y_pred = np.round(model.predict(X_test, num_iteration=model.best_iteration))
        
        precision_list.append(precision_score(y_test, y_pred, pos_label=0))
        f1_list.append(f1_score(y_test, y_pred, pos_label=0))

    if np.mean(f1_list) > 0.9:
        return np.mean(precision_list)
    else:
        return 
    
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=8000)
# %%
#Best parameters

params = {'objective':'binary',
          'seed': 42,
          'metric':'None',
          'boosting':'gbdt',
          'learning_rate': 0.8162682195857298,
          'max_depth': 28,
          'num_leaves': 38,
          'lambda_l1': 0.6781105338379678,
          'lambda_l2': 0.35771025046285154,
          'max_bins': 944,
          'verbosity':-1}

# %%

#GBDT with the best set of hyperparameters
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from metrics import *

y = pd.DataFrame(df['loan_status'])
X = df.drop(y, axis=1)

X_train, X_temp, y_train, y_temp= train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test= train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

train = lgb.Dataset(X_train, y_train)
valid = lgb.Dataset(X_val, y_val, reference=train)

model = lgb.train(params, train, 10000, valid_sets=[valid], feval=precision_metric,  callbacks=[lgb.early_stopping(500)]) 

#Threshold that provides the best balance between FP and FN
y_pred = np.where(model.predict(X_test, num_iteration=model.best_iteration) >= 0.27, 1, 0)


print('Precision',precision_score(y_test, y_pred, pos_label=0))
print('Accuracy',accuracy_score(y_test, y_pred))
print('Recall',recall_score(y_test, y_pred, pos_label=0))


ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize='all', display_labels=['Non default','Default'])

lgb.plot_tree(model)

#Feaure importance based on the gain, i.e 
# Importance based on how much a feature contributes to reduce the loss function
lgb.plot_importance(model, importance_type='gain', title= 'Feature importance GBDT based on Gain')
#%%
#Save model
# model.save_model('GBDT.txt')

#Load model
model = lgb.Booster(model_file='GBDT.txt')  

X_train, X_temp, y_train, y_temp= train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test= train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

y_pred = np.where(model.predict(X_test, num_iteration=model.best_iteration) >= 0.27, 1, 0)
print('Precision',precision_score(y_test, y_pred, pos_label=0))
print('Accuracy',accuracy_score(y_test, y_pred))
print('Recall',recall_score(y_test, y_pred, pos_label=0))
#%%
#Cross_validation
kf = KFold(n_splits=5)
precision_list = []
accuracy_list = []
recall_list = []
for train_index, test_index in kf.split(X):    
    X_train, X_temp = X.iloc[train_index], X.iloc[test_index] 
    y_train, y_temp = y.iloc[train_index], y.iloc[test_index]
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  

    train = lgb.Dataset(X_train, y_train) 
    valid = lgb.Dataset(X_val, y_val, reference=train)
    model = lgb.train(params, train, 10000, feval=precision_metric, valid_sets=[valid], callbacks=[lgb.early_stopping(50)])

    y_pred = np.where(model.predict(X_test, num_iteration=model.best_iteration) >= 0.27, 1, 0)

    precision_list.append(precision_score(y_test, y_pred, pos_label=0))
    accuracy_list.append(accuracy_score(y_test, y_pred))
    recall_list.append(recall_score(y_test, y_pred, pos_label=0))

print('Precision CV',np.mean(precision_list))
print('Accuracy CV',np.mean(accuracy_list))
print('Recall CV',np.mean(recall_list))
#%%
import matplotlib.pyplot as plt

params = {'objective':'binary',
          'seed': 42,
          'metric':'binary_logloss',
          'boosting':'gbdt',
          'learning_rate': 0.8162682195857298,
          'max_depth': 28,
          'num_leaves': 38,
          'lambda_l1': 0.6781105338379678,
          'lambda_l2': 0.35771025046285154,
          'max_bins': 944,
          'verbosity':-1}

#Learning curves
evals = {}
# model = lgb.train(params, train, 1000, feval=precision_metric, valid_sets=[train, valid], callbacks=[lgb.record_evaluation(evals)]) 
model = lgb.train(params, train, 200, valid_sets=[train, valid], callbacks=[lgb.record_evaluation(evals)]) 

ax = lgb.plot_metric(evals, title='Learning curves GBDT (binary_logloss)')

# valid_metric = evals['valid_1']['precision']
# iter = valid_metric.index(max(valid_metric))
valid_metric = evals['valid_1']['binary_logloss']
iter = valid_metric.index(min(valid_metric))


plt.axvline(x=iter, color='r', linestyle='--', label='Min Validation loss')
plt.legend()
plt.show()
# %%

#Model_Reliance

from Model_reliance import *
import lightgbm

y = pd.DataFrame(df['loan_status'])
X = df.drop(y, axis=1)

params = {'objective':'binary',
          'seed': 42,
          'metric':'binary_logloss',
          'boosting':'gbdt',
          'learning_rate': 0.8162682195857298,
          'max_depth': 28,
          'num_leaves': 38,
          'lambda_l1': 0.6781105338379678,
          'lambda_l2': 0.35771025046285154,
          'max_bins': 944,
          'verbosity':-1}

model_reliance = Model_Reliance(X, y)
model_reliance.model_training(lightgbm, lgb=True, params=params)
mr = model_reliance.model_reliance(True)

# %%
