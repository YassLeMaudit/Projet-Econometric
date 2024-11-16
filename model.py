import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data_output/data.csv')

df_long = pd.melt(
    data,
    id_vars=['Mesure (Unité de mesure combinée)', 'Domaine'],
    value_vars=[col for col in data.columns if col.startswith('y_')],
    var_name='year',
    value_name='value'
)


df_long['year'] = df_long['year'].str.replace('y_', '').astype(int)

print(df_long.head())


model = smf.ols(
    formula='value ~ year + C(Q("Mesure (Unité de mesure combinée)")) + C(Domaine)',
    data=df_long
).fit()


model_sum = model.summary()
print(model_sum)


futur_years = np.arange(2022, 2051)
futur_data = pd.DataFrame({
    'year': futur_years,
    'Mesure (Unité de mesure combinée)': ['Emploi dans les activités de l\'économie verte'] * len(futur_years),
    'Domaine': ['Eco-activités'] * len(futur_years)
})

futur_data['value'] = model.predict(futur_data)


print(futur_data)


future_data = futur_data


plt.figure(figsize=(10, 6))


plt.plot(
    future_data['year'],
    future_data['value'],
    label='Prédictions futures',
    linestyle='--'
)

plt.title("Projection des valeurs jusqu'en 2050")
plt.xlabel("Année")
plt.ylabel("Valeurs")
plt.legend()
plt.grid()
plt.show()
