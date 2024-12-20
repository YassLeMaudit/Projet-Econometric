import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from helpers.dataset2_cleaning import clean_dataset
import plotly.graph_objs as go
import plotly.express as px


data = clean_dataset()


ignored_columns = ['Mesure (Unité de mesure combinée)', 'Domaine']

df_long = pd.melt(
    data,
    id_vars=['Mesure (Unité de mesure combinée)', 'Domaine'],
    value_vars=[col for col in data.columns if col not in ignored_columns],
    var_name='year',
    value_name='value'
)

df_long['year'] = df_long['year'].astype(int)

domaines = df_long['Domaine'].unique()

all_predictions = pd.DataFrame()

X = df_long[['year', 'value']]  


X = add_constant(X)


vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]


print(vif_data)


for domaine in domaines:
    domaine_data = df_long[df_long['Domaine'] == domaine]
    model = smf.ols(formula='value ~ year', data=domaine_data).fit()
    futur_years = np.arange(2022, 2051)
    futur_data = pd.DataFrame({
        'year': futur_years,
        'Domaine': domaine
    })
    futur_data['value'] = model.predict(futur_data)
    all_predictions = pd.concat([all_predictions, futur_data], ignore_index=True)

print(all_predictions.head())


# Create a figure
fig = go.Figure()

# Add traces for each 'Domaine'
for domaine in domaines:
    domaine_predictions = all_predictions[all_predictions['Domaine'] == domaine]
    fig.add_trace(go.Scatter(
        x=domaine_predictions['year'],
        y=domaine_predictions['value'],
        mode='lines',
        name=domaine
    ))

# Update layout
fig.update_layout(
    title="Prédictions des emplois par domaine jusqu'en 2050",
    xaxis_title="Année",
    yaxis_title="Nombre d'emplois",
    legend_title="Domaine",
    template="plotly_white"
)

# Show the figure
fig.show()