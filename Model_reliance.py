import pandas as pd
import numpy as np
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
from sklearn.exceptions import DataConversionWarning
from scipy.special import xlogy

warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')


class Model_Reliance:
    def __init__(self, X, y):
        self.y = y
        self.X = X
    
    def loss(self, y ,p):
        return np.mean(-(xlogy(y, p) + xlogy(1 - y, 1 - p)))
    
    def hinge_loss(self, y_true, y_pred):
        return np.mean(np.maximum(0, 1 - y_true * y_pred))
            
    def model_training(self, model=None, lgb=None, params=None, svm=None):
        if not lgb:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=42)
            model.fit(X_train, y_train)
        else:
            X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
            train = lightgbm.Dataset(X_train, y_train)
            valid = lightgbm.Dataset(X_val, y_val, reference=train)
            model_lgb = model.train(params, train, 220, valid_sets=[valid])

        if not svm:
            y_pred = model.predict_proba(X_train)[:, 1] if not lgb else model_lgb.predict(X_train, num_iteration=model_lgb.best_iteration)
        else:
            y_pred = model.decision_function(X_train)

        self.X_train = X_train
        self.y_train = y_train
        self.expected_loss = log_loss(y_train.values, y_pred) if not svm else self.hinge_loss(y_train.values, y_pred)
        self.n = len(X_train)
        self.model = model if not lgb else model_lgb
    

    def loss_column_calculate(self, col, lgb=None, svm=None):
        losses = []
        col_idx = self.X_train.columns.get_loc(col)
        y_train = self.y_train.values
        X_train_values = self.X_train.values


        for index, row in tqdm(self.X_train.reset_index().drop(columns='index').iterrows(), total=self.X_train.shape[0], desc=f"Calculating loss for column {col}"):
            y_true = np.repeat(y_train[index], y_train.shape[0] - 1)
            pairs_temp = np.delete(X_train_values[:, col_idx], index)
            df_noised = np.repeat(row.values.reshape(1, -1), self.n-1, axis=0)
            df_noised[:, col_idx] = pairs_temp

            if not svm:
                prob = self.model.predict_proba(df_noised)[:, 1] if not lgb else self.model.predict(df_noised, num_iteration=self.model.best_iteration)
                losses.append(self.loss(y_true, prob))
            else:
                prob = self.model.decision_function(df_noised)
                losses.append(self.hinge_loss(y_true, prob))
        expected_loss_noised = np.mean(losses)

        return expected_loss_noised/self.expected_loss


    def model_reliance(self, lgb=None, svm=None):
        model_reliance = pd.DataFrame(columns=['Model_Reliance'], index=self.X_train.columns)
        for col in tqdm(self.X_train.columns, desc="Calculating losses for all columns"):
            if lgb:
                model_reliance.loc[col, 'Model_Reliance'] = self.loss_column_calculate(col, lgb=True)
            elif svm:
                model_reliance.loc[col, 'Model_Reliance'] = self.loss_column_calculate(col, svm=True)
            else:   
                model_reliance.loc[col, 'Model_Reliance'] = self.loss_column_calculate(col)

        model_reliance.to_csv(f'Model_Reliance_{str(self.model)}.csv')

        return model_reliance

