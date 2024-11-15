import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


with open('data_output\dataset2_preanalysis.csv',encoding='UTF-8') as f:
    dataset2 = pd.read_csv(f)


# Fit regression model (using the natural log of one of the regressors)
results = smf.ols('y_2008 ~ y_2009 + np.log(y_2010)', data=dataset2).fit()

# Inspect the results
print(results.summary())