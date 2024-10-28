
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

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from statsmodels.stats.outliers_influence import variance_inflation_factor 


y = pd.DataFrame(df['loan_status'])
X = df.drop(y, axis=1)
X = df[['loan_percent_income', 'loan_grade', 'log_person_income', 'person_home_ownership', 'loan_intent']]
vif_data = pd.DataFrame() 
vif_data["feature"] = X.columns 
  
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))] 
  
print(vif_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#%%

#Training a baseline model (Logistic Regression) without any regularization
model = LogisticRegression(max_iter=10000, random_state=42, penalty=None)
model.fit(X_train, y_train)
y_pred = np.round(model.predict_proba(X_test)[:, 1])
    
feature_importance = pd.DataFrame(model.coef_[0], index = model.feature_names_in_, columns=['Coefficients'])
feature_importance.sort_values('Coefficients', ascending=True).plot.barh(grid=True, title='Coefficient Levels for All Features, Logistic Regression', xlabel='Coefficient level', ylabel='Features', legend=False)

print('Accuracy :',  accuracy_score(y_test, y_pred))
print('Precision:',  precision_score(y_test, y_pred, pos_label=0))
print('Recall:', recall_score(y_test, y_pred, pos_label=0))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize='all', display_labels=['Non default','Default'])
#%%
import pickle 

#Save model
# pickle.dump(model, open('LogisticRegression.sav', 'wb'))


#Load model
model = pickle.load(open('LogisticRegression.sav', 'rb'))

X = df[['loan_percent_income', 'loan_grade', 'log_person_income', 'person_home_ownership', 'loan_intent']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

result = model.score(X_test, y_test)
print(result)

#%%
#Cross-validation to see how well the model generalise 
kf = KFold(n_splits=10)
precision_list = []
accuracy_list = []
recall_list = []
for i, (train_index, test_index) in enumerate(kf.split(X)):    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)

    y_pred = np.round(model.predict_proba(X_test)[:, 1])

    precision_list.append(precision_score(y_test, y_pred, pos_label=0))
    accuracy_list.append(accuracy_score(y_test, y_pred))
    recall_list.append(recall_score(y_test, y_pred, pos_label=0))

print('Precision CV',np.mean(precision_list))
print('Accuracy CV',np.mean(accuracy_list))
print('Recall CV',np.mean(recall_list))
#%%

import matplotlib.pyplot as plt
import seaborn as sns

#Plot the logistic function
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

z = np.dot(X_test, model.coef_.T) + model.intercept_

probabilities = 1 / (1 + np.exp(-z))

z = z.flatten()
probabilities = probabilities.flatten()

sorted_indices = np.argsort(z)
z_sorted = z[sorted_indices]
probabilities_sorted = probabilities[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]


np.random.seed(42)
subset_indices = np.random.choice(len(z_sorted), size=1500, replace=False)

z_subset = z_sorted[subset_indices]
y_pred_subset = y_pred_sorted[subset_indices]

plt.figure(figsize=(8, 6))
plt.plot(z_sorted, probabilities_sorted, color='red', label='Logistic Regression Model')

sns.scatterplot(x=z_subset, y=y_pred_subset, hue=y_pred_subset)

plt.title('Logistic Regression Model')
plt.xlabel('z')
plt.ylabel('Probability')
plt.grid(True)
plt.legend(loc='best')
plt.show()
#%%

#Model Reliance

from Model_reliance import *
from sklearn.linear_model import LogisticRegression


y = pd.DataFrame(df['loan_status'])
X = df[['loan_percent_income', 'loan_grade', 'log_person_income', 'person_home_ownership', 'loan_intent']]

model = LogisticRegression(max_iter=10000, random_state=42, penalty=None)

model_reliance = Model_Reliance(X, y)
model_reliance.model_training(model=model)
mr = model_reliance.model_reliance()

# %%
