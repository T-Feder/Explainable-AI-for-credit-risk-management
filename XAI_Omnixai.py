#%%
import pandas as pd
import os
os.chdir('C:/Users/33781/Downloads/M2 FTD S2/Applied Machine Learning')
import numpy as np
from data_processing import *
from metrics import *
#%%

df = pd.read_csv('credit_risk_dataset.csv')
data = data_processing(df)
data()
df = data.df
y = pd.DataFrame(df['loan_status'])
X = df.drop(y, axis=1)
#%%

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from omnixai.data.tabular import Tabular
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.explainers.tabular import TabularExplainer
from omnixai.visualization.dashboard import Dashboard


tabular_data = Tabular(
    data=df,
    target_column='loan_status'
)
# %%
transformer = TabularTransform().fit(tabular_data)
class_names = transformer.class_names
x = transformer.transform(tabular_data)
X_train, X_temp, y_train, y_temp= train_test_split(x[:, :-1], x[:, -1], test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test= train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

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

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric=precision_metric_classifier, callbacks=[lgb.early_stopping(50)])
train = transformer.invert(X_train)
test = transformer.invert(X_test)
# %%

preprocess = lambda z: transformer.transform(z)
explainers=["shap", "lime", "ale", "shap_global"]

explainer = TabularExplainer(
    explainers=explainers,
    mode="classification",
    data=train,
    model=model,
    preprocess=preprocess
)

local_explanations = explainer.explain(X=test[:20])
global_explanations = explainer.explain_global()

# %%
from omnixai.explainers.prediction import PredictionAnalyzer

analyzer = PredictionAnalyzer(
    mode="classification",
    test_data=test,
    test_targets=y_test,
    model=model,
    preprocess=preprocess
)
prediction_explanations = analyzer.explain()
# %%

dashboard = Dashboard(
    instances=test[:20],
    local_explanations=local_explanations,
    global_explanations=global_explanations,
    prediction_explanations=prediction_explanations,
    class_names=class_names,
    explainer=explainer
)

dashboard.show()
# %%
