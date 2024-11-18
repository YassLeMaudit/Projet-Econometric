import pandas as pd

# Load the CSV files
df_emplois = pd.read_csv('data_output/data_emplois_cleaned.csv')
df_gdp = pd.read_csv('data_output/gdp.csv')

# Inspect the columns
print("df_emplois columns:", df_emplois.columns)
print("df_gdp columns:", df_gdp.columns)

# Ensure both DataFrames have the same columns
for col in df_gdp.columns:
    if col not in df_emplois.columns:
        df_emplois[col] = ''

for col in df_emplois.columns:
    if col not in df_gdp.columns:
        df_gdp[col] = ''

# Reorder columns to match df_emplois
df_gdp = df_gdp[df_emplois.columns]

# Concatenate the DataFrames
value = pd.concat([df_emplois, df_gdp], ignore_index=True)

# Save the combined DataFrame to a new CSV file
with open(r'data_output/dataset_gdp1.csv', 'w', newline='', encoding='UTF-8') as f:
    value.to_csv(f, index=False)