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

from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score,  accuracy_score, recall_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from metrics import *
from statsmodels.stats.outliers_influence import variance_inflation_factor 

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X = df[['loan_percent_income', 'loan_grade', 'log_person_income', 'person_home_ownership', 'loan_intent']]
vif_data = pd.DataFrame() 
vif_data["feature"] = X.columns 
  
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                          for i in range(len(X.columns))] 
  
print(vif_data)

# Scaling features since SVMs are sensitive to the scale of the data
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)
#%%

#Training SVM 
svm_model = LinearSVC(max_iter=10000, random_state=42)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

feature_importance = pd.DataFrame(svm_model.coef_[0], index = svm_model.feature_names_in_, columns=['Coefficients'])
feature_importance.sort_values('Coefficients', ascending=True).plot.barh(grid=True, title='Coefficient Levels for All Features, SVM', xlabel='Coefficient Level', ylabel='Features', legend=False)

print('Precision',precision_score(y_test, y_pred, pos_label=0))
print('Accuracy',accuracy_score(y_test, y_pred))
print('Recall',recall_score(y_test, y_pred, pos_label=0))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize='all', display_labels=['Non default','Default'])
#%%

import pickle 

#Save model
# pickle.dump(svm_model, open('SVM.sav', 'wb'))


#Load model
svm_model = pickle.load(open('SVM.sav', 'rb'))

scaler = StandardScaler()
X = df[['loan_percent_income', 'loan_grade', 'log_person_income', 'person_home_ownership', 'loan_intent']]
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

result = svm_model.score(X_test, y_test)
print(result)

#%%

#Cross-validation to see how-well the model generalise
kf = KFold(n_splits=10)
precision_list = []
accuracy_list = []
recall_list = []
for train_index, test_index in kf.split(X_scaled):    
    X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index] 
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    precision_list.append(precision_score(y_test, y_pred, pos_label=0))
    accuracy_list.append(accuracy_score(y_test, y_pred))
    recall_list.append(recall_score(y_test, y_pred, pos_label=0))

print('Precision CV',np.mean(precision_list))
print('Accuracy CV',np.mean(accuracy_list))
print('Recall CV',np.mean(recall_list))

#%%


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC

#Plot the plane

#We added loan_int_rate only for visualization
X = df[['loan_percent_income', 'loan_grade', 'log_person_income', 'person_home_ownership', 'loan_intent', 'loan_int_rate']]
y = df['loan_status']
X = X[['loan_percent_income', 'loan_int_rate']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                   test_size=0.1, stratify=y,
                                   random_state=42)

svc_model = SVC(kernel='linear', random_state=42)
svc_model.fit(X_train, y_train)




w = svc_model.coef_[0]
b = svc_model.intercept_[0]

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

Z = (w[0] * xx + w[1] * yy + b)
Z = Z.reshape(xx.shape)



np.random.seed(42)
selected_0 = np.random.choice(np.where(y_train == 0)[0], round(len(X_train) * 0.025), replace=False)
selected_1 = np.random.choice(np.where(y_train == 1)[0], round(len(X_train) * 0.025), replace=False)

plt.figure(figsize=(10, 8))
plt.scatter(X_train[selected_0, 0], X_train[selected_0, 1], c='blue', s=6,  label='0')
plt.scatter(X_train[selected_1, 0], X_train[selected_1, 1], c='orange', s=6,  label='1')
plt.contour(xx, yy, Z, colors=['k'], levels=[0], linestyles=['-'], label='Hyperplan')
plt.contour(xx, yy, Z + 1, colors=['r'], levels=[0], linestyles=['--'], label='Marge +1')
plt.contour(xx, yy, Z - 1, colors=['r'], levels=[0], linestyles=['--'], label='Marge -1')


selected_indices = np.concatenate([selected_0, selected_1])
selected_support_indices = np.intersect1d(svc_model.support_, selected_indices)
selected_support_vectors = svc_model.support_vectors_[np.isin(svc_model.support_, selected_support_indices)]

if len(selected_support_vectors) > 100:
    selected_support_vectors = selected_support_vectors[np.random.choice(len(selected_support_vectors), 100, replace=False)]

plt.scatter(selected_support_vectors[:, 0], selected_support_vectors[:, 1], s=50, facecolors='none', edgecolors='k', label='Selected Support Vectors', alpha=0.5)

plt.xlabel('loan_percent_income')
plt.ylabel('loan_int_rate')
plt.title('Support Vector Machines plane')
plt.legend()
plt.show()


#%%

#Model Reliance
from Model_reliance import *
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

y = pd.DataFrame(df['loan_status'])
X = df[['loan_percent_income', 'loan_grade', 'log_person_income', 'person_home_ownership', 'loan_intent']]


scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

#Hinge-loss necessitates +1/-1 outputs for good handling
y=pd.DataFrame(np.where(y==0, 1,-1), columns=['loan_status'])
svm_model = LinearSVC(max_iter=10000, random_state=42)

model_reliance = Model_Reliance(X_scaled, y)
model_reliance.model_training(model=svm_model, svm=True)
mr = model_reliance.model_reliance(svm=True)

# %%
