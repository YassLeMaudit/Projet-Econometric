import pandas as pd
import statsmodels.formula as smf
import numpy as np

df = pd.read_csv('data_initial/Data.csv')
data_transpose = df.T
data_transpose.to_csv('Data_transpose.csv', index= False)