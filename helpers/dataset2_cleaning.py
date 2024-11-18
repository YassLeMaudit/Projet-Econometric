import pandas as pd


def clean_dataset():
    with open(r'data_initial\dataset2_emplois.csv',encoding='UTF-8') as f:
        dataset2 = pd.read_csv(f)

    def remove_space(dataframe):
        for column in dataframe.columns[2::]:
            dataframe[column] = dataframe[column].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float)
        return dataframe

    df_final = remove_space(dataset2)
    # Calculate the sum of each column from the 2nd column to the end
    sum_column = df_final.columns[2::]
    for column in sum_column:
        print(f"Sum of {column}: {df_final[column].sum()}")
    print(df_final[sum_column].applymap(lambda x: isinstance(x, str)))
    df_final[sum_column] = df_final[sum_column].apply(pd.to_numeric, errors='coerce').fillna(0)

    total_row = df_final[sum_column].sum().to_frame().T

    # Add 'Total' and 'Tout' to the first two columns of the total row
    total_row.insert(0, df_final.columns[1], 'Tout')
    total_row.insert(0, df_final.columns[0], 'Total')
    df_final = pd.concat([df_final, total_row], ignore_index=True)

    # Rename the columns with the years to remove decimals
    rename_dict = {f'{year},00': str(year) for year in range(2008, 2022)}
    df_final.rename(columns=rename_dict, inplace=True)

    # Remove 'Total Général rows' and values in % rows
    df_final.drop(df_final[(df_final['Domaine'] == 'Total général') | (df_final['Domaine'] == 'Évolution annuelle en %')].index, inplace=True)
    print(df_final)
    # Save the dataset to a csv file to make tests
    with open(r'data_output\data_emplois_cleaned.csv', 'w', newline='', encoding='UTF-8') as f:
        df_final.to_csv(f, index=False)
    print('Dataset cleaned and saved to data_output/data_emplois_cleaned.csv')
    return df_final

clean_dataset()