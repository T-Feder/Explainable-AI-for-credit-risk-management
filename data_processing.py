
#%%
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

class data_processing:

    def __init__(self, df):
        self.df = df

    def __call__(self):
        df = self.df

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
        encoders = {}
        for encoder in encoders_string:
            le = LabelEncoder()
            df[encoder] = le.fit_transform(df[encoder])
            encoders[encoder] = le
        


        #KNN imputation, we normalize first as it's a measure based on Eucledian distance
        #KNN permits to keep the original distribution even after imputation
        scaler = MinMaxScaler()
        impute = KNNImputer()

        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        df = pd.DataFrame(impute.fit_transform(df), columns=df.columns)
        df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)


        self.df = df
        self.categorical_cols = encoders_string
        self.encoders = encoders


# %%

# #Example of usage
# df = pd.read_csv('credit_risk_dataset.csv')

# data = data_processing(df)
# data()
# df = data_processing.df
