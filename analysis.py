import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


with open('data_output\dataset2_preanalysis.csv',encoding='UTF-8') as f:
    dataset2 = pd.read_csv(f)


dat = sm.datasets.get_rdataset("Guerry", "HistData").data

# Fit regression model (using the natural log of one of the regressors)
results = smf.ols('2009,00 ~ 2008,00 + np.log(Domaine)', data=dataset2).fit()

# Inspect the results
print(results.summary())