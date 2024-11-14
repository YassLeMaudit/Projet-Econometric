import pandas as pd


with open('data_initial/Data.csv') as f:
    scenarios = pd.read_csv(f)

S1 = scenarios[scenarios['Scenario'] == 'S1'].transpose()
S2 = scenarios[scenarios['Scenario'] == 'S2'].transpose()
S3 = scenarios[scenarios['Scenario'] == 'S3'].transpose()
S4 = scenarios[scenarios['Scenario'] == 'S4'].transpose()

S1.drop('Scenario', inplace=True)
S2.drop('Scenario', inplace=True)
S3.drop('Scenario', inplace=True)
S4.drop('Scenario', inplace=True)

S1.to_csv('data_output/S1.csv',index=False)
S2.to_csv('data_output/S2.csv',index=False)
S3.to_csv('data_output/S3.csv',index=False)
S4.to_csv('data_output/S4.csv',index=False)
