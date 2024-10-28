#%%
import pandas as pd
import os
os.chdir('C:/Users/33781/Downloads/M2 FTD S2/Applied Machine Learning')
import numpy as np
#%%

from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelEncoder
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv('credit_risk_dataset.csv')


#We remake the percent_income to add precision 
df['loan_percent_income'] = df['loan_amnt']/df['person_income']


df['loan_int_rate'] /= 100
df['loan_status'] = df['loan_status'].astype(float)


#We remove potential false observations that may add noise
outliers_values = [123, 144]
df = df.loc[(~df['person_age'].isin(outliers_values)) & (df['person_emp_length'] != 123)]



#Feature engineering
df['exp_age'] = df['person_emp_length'] / df['person_age']
df['credit_hist_age'] = df['cb_person_cred_hist_length'] / df['person_age']
df['exp_credit_hist'] = df['person_emp_length']/df['cb_person_cred_hist_length']
df['interest'] = df['loan_int_rate'] * df['loan_amnt']
df['age_squared'] = df['person_age']**2
df['exp_squared'] = df['person_emp_length']**2


#We set a log scale for the income, loan_amount and interests
df['person_income']= np.log(df['person_income'])
df['loan_amnt'] = np.log(df['loan_amnt'])
df['interest'] = np.log(df['interest'] )
df.rename(columns={'person_income':'log_person_income', 'loan_amnt':'log_loan_amnt', 'interest':'log_interest'},inplace=True)



#Converting categorical into dummies
encoders_string = [col for col in df.columns if df[col].dtype == 'object']
for encoder in encoders_string:
    exec(f'{encoder} = LabelEncoder()')
encoded_cols = pd.DataFrame([pd.Series(globals()[encoder].fit_transform(df[encoder]), index=df.index) for encoder in encoders_string]).T
df[encoders_string] = encoded_cols


#Distribution of cols with missing values before imputation
#Readapt the titles to show the before after imputation
sns.displot(data=df, x='person_emp_length', kde=True, stat='density')
sns.displot(data=df, x='loan_int_rate', kde=True, stat='density')
plt.show()


#KNN imputation, we normalize first as it's a measure based on Eucledian distance
#KNN permits to keep the original distribution even after imputation
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

scaler = MinMaxScaler()
impute = KNNImputer()

df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df = pd.DataFrame(impute.fit_transform(df), columns=df.columns)
df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)

#Distribution after imputation
sns.displot(data=df, x='person_emp_length', kde=True, stat='density')
sns.displot(data=df, x='loan_int_rate', kde=True, stat='density')
plt.show()
#%%

#Baseline model with all the features

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score
from metrics import *

y = pd.DataFrame(df['loan_status'])
X = df.drop(y, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = LogisticRegression(max_iter=10000, random_state=42, penalty=None)
model.fit(X_train, y_train)
y_pred = np.where(model.predict_proba(X_test)[:, 1] >= 0.6, 1, 0)
    
feature_importance = pd.DataFrame(model.coef_[0], index = model.feature_names_in_)
feature_importance.sort_values(0, ascending=False)

#We set the posisitve class to 0 as it corresponds to no default case
print('accuracy :',  accuracy_score(y_test, y_pred),
      'precision_score :',  precision_score(y_test, y_pred, pos_label=0),
      'pnpv :', precision_npv_score(y_test, y_pred))

#----------------------------------------------------------------------------------

#Cross-validation to see how well our model generalise 
kf = KFold(n_splits=10)
precision_list = []
accuracy_list = []
pnvp_list = []
for i, (train_index, test_index) in enumerate(kf.split(X)):    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)

    y_pred = np.where(model.predict_proba(X_test)[:, 1] >= 0.6, 1, 0)

    precision_list.append(precision_score(y_test, y_pred, pos_label=0))
    accuracy_list.append(accuracy_score(y_test, y_pred))
    pnvp_list.append(precision_npv_score(y_test, y_pred))

print(np.mean(precision_list))
print(np.mean(accuracy_list))
print(np.mean(pnvp_list))
#%%

from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import precision_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV
from metrics import *
from sklearn.metrics import accuracy_score, precision_score



X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Scaling features since SVMs are sensitive to the scale of the data
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

svm_model = LinearSVC(max_iter=10000, random_state=42)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)


print(precision_score(y_test, y_pred, pos_label=0))
print(accuracy_score(y_test, y_pred))
print(precision_npv_score(y_test, y_pred))


kf = KFold(n_splits=10)
precision_list = []
accuracy_list = []
pnvp_list = []
for train_index, test_index in kf.split(X_scaled):    
    X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index] 
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    precision_list.append(precision_score(y_test, y_pred, pos_label=0))
    accuracy_list.append(accuracy_score(y_test, y_pred))
    pnvp_list.append(precision_npv_score(y_test, y_pred))

print(np.mean(precision_list))
print(np.mean(accuracy_list))
print(np.mean(pnvp_list))

#----------------------------------------------------------------------------------

param_grid_linear = {
    'C': [0.01, 0.1, 1, 10, 100]  
}

# Fine-Tuning SVM with linear Kernel 
grid_search_linear = GridSearchCV(SVC(kernel='linear', max_iter=10000, random_state=42), param_grid_linear,
                                  scoring={'accuracy': 'accuracy',
                                           'precision': make_scorer(precision_score, pos_label=0)},
                                  refit='accuracy',  
                                  cv=6, verbose=1)

# Fit GridSearchCV
grid_search_linear.fit(X_scaled, y)

# Best parameters and best score
print('Best parameters for linear SVM:', grid_search_linear.best_params_)
print('Best score for linear SVM:', grid_search_linear.best_score_)


#----------------------------------------------------------------------------------

#Fine-Tuning SVM with rbf Kernel 
param_grid_rbf = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10, 100]  
}

# Create GridSearchCV object for SVC with RBF kernel
grid_search_rbf = GridSearchCV(SVC(kernel='rbf', max_iter=10000, random_state=42), param_grid_rbf, 
                               scoring={'accuracy': 'accuracy', 
                                        'precision': make_scorer(precision_score, pos_label=0)},
                               refit='accuracy',  
                               cv=6, verbose=1)

# Fit GridSearchCV
grid_search_rbf.fit(X_scaled, y)

# Best parameters and best score
print('Best parameters for RBF SVM:', grid_search_rbf.best_params_)
print('Best score for RBF SVM:', grid_search_rbf.best_score_)

#----------------------------------------------------------------------------------


svm_model = SVC(max_iter=10000, C=10, random_state=42, probability=True)
svm_model.fit(X_train, y_train)

y_pred = np.where(svm_model.predict_proba(X_test)[:, 1] >= 0.6, 1, 0)


print(precision_score(y_test, y_pred, pos_label=0))
print(accuracy_score(y_test, y_pred))
print(precision_npv_score(y_test, y_pred))

#----------------------------------------------------------------------------------

kf = KFold(n_splits=10)
precision_list = []
accuracy_list = []
pnvp_list = []
for train_index, test_index in kf.split(X_scaled):    
    X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index] 
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    svm_model.fit(X_train, y_train)

    y_pred = np.where(svm_model.predict_proba(X_test)[:, 1] >= 0.6, 1, 0)

    precision_list.append(precision_score(y_test, y_pred, pos_label=0))
    accuracy_list.append(accuracy_score(y_test, y_pred))
    pnvp_list.append(precision_npv_score(y_test, y_pred))

print(np.mean(precision_list))
print(np.mean(accuracy_list))
print(np.mean(pnvp_list))


#%%
#GBDT baseline with all features

import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score
from metrics import *

y = pd.DataFrame(df['loan_status'])
X = df.drop(y, axis=1)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.1, random_state=42)
train = lgb.Dataset(X_train, label = y_train)

params = {'objective':'binary',
          'seed': 42,
          'metric':'binary_logloss',
          }

model = lgb.train(params, train, 1000)
y_pred = np.round(model.predict(X_test))

print(precision_score(y_test, y_pred, pos_label=0))
print(accuracy_score(y_test, y_pred))
print(precision_npv_score(y_test, y_pred))
lgb.plot_importance(model)

#----------------------------------------------------------------------------------

kf = KFold(n_splits=10)
precision_list = []
accuracy_list = []
pnvp_list = []
for i, (train_index, test_index) in enumerate(kf.split(X)):    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    train = lgb.Dataset(X_train, label = y_train) 
    model = lgb.train(params, train, 1000)

    y_pred = np.round(model.predict(X_test))

    precision_list.append(precision_score(y_test, y_pred, pos_label=0))
    accuracy_list.append(accuracy_score(y_test, y_pred))
    pnvp_list.append(precision_npv_score(y_test, y_pred))

print(np.mean(precision_list))
print(np.mean(accuracy_list))
print(np.mean(pnvp_list))

#%%
#Fine-tuning to find the best hyperparameters with cv + validation data on strat sets

import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from metrics import *
from datetime import datetime

y = pd.DataFrame(df['loan_status'])
X = df.drop(y, axis=1)


def objective(trial):
    print(datetime.now())

    lr = trial.suggest_float('learning_rate', 1e-5, 1, log=True)
    max_depth = trial.suggest_int('max_depth', 1, 100)
    num_leaves = trial.suggest_int('num_leaves', 31, 1000)
    lambda_1 = trial.suggest_float('lambda_1', 0, 1)
    lambda_2 = trial.suggest_float('lambda_2', 0, 1)
    max_bins = trial.suggest_int('max_bins', 255, 1024)
    
    params={'objective':'binary',
            'seed':42,
            'metric':'binary_logloss',
            'device':'cpu',
            'learning_rate':lr,
            'max_depth':max_depth,
            'num_leaves':num_leaves,
            'max_bins':max_bins,
            'lambda_l1':lambda_1,
            'lambda_l2':lambda_2,
            'verbosity':-1}
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = []
    for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_temp, y_temp = X.iloc[test_index], y.iloc[test_index]
        X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
        del X_temp, y_temp
        
        train = lgb.Dataset(X_train, label=y_train)
        valid = lgb.Dataset(X_valid, label=y_valid, reference=train)
        
        model = lgb.train(params, train, 10000, valid_sets=[valid], callbacks=[lgb.early_stopping(50)])
        
        y_pred = np.round(model.predict(X_test, num_iteration=model.best_iteration))
        
        score.append(precision_npv_score(y_test, y_pred))
    
    return np.mean(score)

study = optuna.create_study(sampler=optuna.samplers.GPSampler(n_startup_trials=400), direction='maximize')
study.optimize(objective, n_trials=2000, n_jobs=2) #Test with 3 to see if better

# %%
#Best parameters with strat

params = {'objective':'binary',
          'seed': 42,
          'metric':'binary_logloss',
          'boosting':'gbdt',
          'learning_rate':0.01,
          'lambda_l1': 0.63, 
          'lambda_l2': 0.9999999999999999,
          'max_depth': 20,
          'num_leaves': 31,
          'max_bins': 1023,
          'verbosity':-1}

# %%
#Best model without strat

import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, log_loss
from metrics import *

y = pd.DataFrame(df['loan_status'])
X = df.drop(y, axis=1)

X_train, X_temp, y_train, y_temp= train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test= train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

train = lgb.Dataset(X_train, y_train)
valid = lgb.Dataset(X_val, y_val, reference=train)

model = lgb.train(params, train, 10000, valid_sets=[valid], callbacks=[lgb.early_stopping(50)]) 
y_pred = np.where(model.predict(X_test, num_iteration=model.best_iteration) >= 0.6, 1, 0)

print(precision_score(y_test, y_pred, pos_label=0))
print(accuracy_score(y_test, y_pred))
print(precision_npv_score(y_test, y_pred))
lgb.plot_importance(model)


#----------------------------------------------------------------------------------

kf = KFold(n_splits=5)
precision_list = []
accuracy_list = []
pnvp_list = []
for train_index, test_index in kf.split(X):    
    X_train, X_temp = X.iloc[train_index], X.iloc[test_index] 
    y_train, y_temp = y.iloc[train_index], y.iloc[test_index]
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  

    train = lgb.Dataset(X_train, y_train) 
    valid = lgb.Dataset(X_val, y_val, reference=train)
    model = lgb.train(params, train, 10000, valid_sets=[valid], callbacks=[lgb.early_stopping(50)])

    y_pred = np.where(model.predict(X_test, num_iteration=model.best_iteration) >= 0.6, 1, 0)

    precision_list.append(precision_score(y_test, y_pred, pos_label=0))
    accuracy_list.append(accuracy_score(y_test, y_pred))
    pnvp_list.append(precision_npv_score(y_test, y_pred))

print(np.mean(precision_list))
print(np.mean(accuracy_list))
print(np.mean(pnvp_list))

#----------------------------------------------------------------------------------

evals = {}
model = lgb.train(params, train, 10000, valid_sets=[train, valid], callbacks=[lgb.record_evaluation(evals)]) 
lgb.plot_metric(evals, title='Learning curves (GBDT without strat)')

#%%
#Model reliance
import lightgbm as lgb
from Model_reliance import *

y = pd.DataFrame(df['loan_status'])
X = df.drop(y, axis=1)


model_reliance = Model_Reliance(X, y)
model_reliance.model_training(lightgbm, lgb=True, params=params)
mr = model_reliance.model_reliance(lgb=True)

#----------------------------------------------------------------------------------

#Model reliance paralellized
from Model_reliance_paralellized import *

model_reliance = Model_Reliance(X, y)
model_reliance.model_training(lightgbm, lgb=True, params=params)
reliance_df = model_reliance.model_reliance(lgb=True)

# %%
#X-AI

import lightgbm as lgb
import xai
from sklearn.model_selection import train_test_split


y = pd.DataFrame(df['loan_status'])
X = df.drop(y, axis=1)


X_train, X_temp, y_train, y_temp= train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test= train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

train = lgb.Dataset(X_train, y_train)
valid = lgb.Dataset(X_val, y_val, reference=train)

model = lgb.train(params, train, 10000, valid_sets=[valid], callbacks=[lgb.early_stopping(50)]) 
prob = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = np.where(prob >= 0.6, 1, 0)



for col in encoders_string:
    encoder = globals()[col]
    df[col] = encoder.inverse_transform(df[col])



# Data visualization
xai.imbalance_plot(df, 'person_home_ownership', 'loan_status')


#Model specificity classification
# xai.metrics_plot(y_test, prob, df = X_test, cross_cols=['person_home_ownership', 'loan_grade'], categorical_cols=encoders_string)



# %%
