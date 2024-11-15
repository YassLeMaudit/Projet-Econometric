import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


with open('data_initial\dataset2_emplois.csv',encoding='UTF-8') as f:
    dataset2 = pd.read_csv(f)

results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dataset2).fit()