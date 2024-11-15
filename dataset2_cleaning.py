import pandas as pd


with open('data_initial\dataset2_emplois.csv',encoding='UTF-8') as f:
    dataset2 = pd.read_csv(f)



def remove_space(dataframe):
    for column in dataframe.columns[2::]:
        dataframe[column] = dataframe[column].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float)
    return dataframe


df_final = remove_space(dataset2)

# Calculate the sum of each column from the 2nd column to the end
sum_column = df_final.columns[2::]
total_row = df_final[sum_column].sum().to_frame().T

# Add 'Total' and 'Tout' to the first two columns of the total row
total_row.insert(0, df_final.columns[1], 'Tout')
total_row.insert(0, df_final.columns[0], 'Total')
df_final = pd.concat([df_final, total_row], ignore_index=True)

# Rename the columns with the years to remove decimals
rename_dict = {f'{year},00': str(year) for year in range(2008, 2022)}
df_final.rename(columns=rename_dict, inplace=True)


# Save the dataset to a csv file to make tests
# with open('data_output\dataset2_preanalysis.csv', 'w', newline='', encoding='UTF-8') as f:
#     df_final.to_csv(f, index=False)