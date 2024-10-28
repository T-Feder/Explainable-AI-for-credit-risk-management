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
y = pd.DataFrame(df['loan_status'])
X = df.drop(y, axis=1)
#%%

#LightGBM
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from metrics import * 

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


X_train, X_temp, y_train, y_temp= train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test= train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


model = lgb.LGBMClassifier(num_iteration=10000,
    objective=params['objective'],
    random_state=params['seed'],
    metric=params['metric'],
    boosting_type=params['boosting'],
    learning_rate=params['learning_rate'],
    reg_alpha=params['lambda_l1'],  
    reg_lambda=params['lambda_l2'],  
    max_depth=params['max_depth'],
    num_leaves=params['num_leaves'],
    max_bin=params['max_bins'],
    verbosity=params['verbosity']
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(500)], eval_metric=precision_metric_classifier)
y_pred = pd.Series(np.where(model.predict(X_test, num_iteration=model.best_iteration_) >= 0.27, 1, 0), index=X_test.index)
#%%

#SVM 
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[['loan_percent_income', 'loan_grade', 'log_person_income', 'person_home_ownership', 'loan_intent']]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

model = LinearSVC(max_iter=10000, random_state=42)
model.fit(X_train, y_train)

y_pred = pd.Series(model.predict(X_test), index=X_test.index)
#%%

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


X = df[['loan_percent_income', 'loan_grade', 'log_person_income', 'person_home_ownership', 'loan_intent']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = LogisticRegression(max_iter=10000, random_state=42, penalty='none')
model.fit(X_train, y_train)
y_pred = pd.Series(np.round(model.predict_proba(X_test)[:, 1]), index=X_test.index)
#%%
from shapash import SmartExplainer

xpl = SmartExplainer(model=model)
xpl.compile(x=X_train)
xpl.compile(x=X_test, y_pred=y_pred, y_target=y_test)

# app = xpl.run_app()

# %%
import pickle


# Save SmartExplainer object
with open("smart_explainer_svm.pkl", 'wb') as file:
    pickle.dump(xpl, file)
# %%
import pickle
from shapash import SmartExplainer

#Load SmartExplainer object
with open("smart_explainer_svm.pkl", 'rb') as file:
    xpl = pickle.load(file)

app = xpl.run_app()
# %%
