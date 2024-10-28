#%%
import pandas as pd
import os
os.chdir('C:/Users/33781/Downloads/M2 FTD S2/Applied Machine Learning')
from ydata_profiling import ProfileReport
import seaborn as sns
import numpy as np
#%%

df = pd.read_csv('credit_risk_dataset.csv')

#%%
#I didn't analysed exp_age 

df['person_income']= np.log(df['person_income'])
df['loan_amnt'] = np.log(df['loan_amnt'])
df.rename(columns={'person_income':'log_person_income', 'loan_amnt':'log_loan_amnt'},inplace=True)
profile = ProfileReport(df)
profile.to_file('wagesdata.html', silent=False)
