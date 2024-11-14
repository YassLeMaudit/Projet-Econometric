import pandas as pd


with open('Data/Data.csv') as f:
    scenarios = pd.read_csv(f)

S1 = scenarios[scenarios['Scenario'] == 'S1'].transpose()
S2 = scenarios[scenarios['Scenario'] == 'S2'].transpose()
S3 = scenarios[scenarios['Scenario'] == 'S3'].transpose()
S4 = scenarios[scenarios['Scenario'] == 'S4'].transpose()

S1.drop('Scenario', inplace=True)
S2.drop('Scenario', inplace=True)
S3.drop('Scenario', inplace=True)
S4.drop('Scenario', inplace=True)

