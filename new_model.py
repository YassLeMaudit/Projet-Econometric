import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Charger les données
data = pd.read_csv('data_output/dataset_gdp.csv')

# Restructurer les données d'emploi
data_long = data.melt(
    id_vars=["Mesure (Unité de mesure combinée)", "Domaine"],
    var_name="Année",
    value_name="Nombre d'emplois"
)
data_long["Année"] = pd.to_numeric(data_long["Année"], errors="coerce")
data_long = data_long.dropna()

# Supprimer les doublons dans data_long
data_long = data_long.drop_duplicates(subset=["Année", "Domaine"])

# Extraire et restructurer les paramètres
param_data = data.iloc[42:64]
params_long = param_data.melt(
    id_vars=["Mesure (Unité de mesure combinée)", "Domaine"],
    var_name="Année",
    value_name="Valeur"
)
params_long["Année"] = pd.to_numeric(params_long["Année"], errors="coerce")

# Renommer la colonne Domaine dans params_long pour éviter les confusions
params_long = params_long.rename(columns={"Domaine": "Paramètre"})

# Fusionner les données
merged_data = pd.merge(
    data_long,
    params_long,
    on="Année",
    how="inner"
)

# Renommer les colonnes pour clarification
merged_data = merged_data.rename(columns={
    "Domaine": "Domaine_emploi",  # Domaine réel des emplois
    "Paramètre": "Nom_paramètre"  # Nom des paramètres fusionnés
})

# Vérifier le résultat
print("Colonnes de merged_data :", merged_data.columns)
print(merged_data.head())

# Supprimer les doublons si nécessaire
merged_data = merged_data.drop_duplicates(subset=["Année", "Domaine_emploi"])
print("Taille de merged_data après nettoyage :", merged_data.shape)

# Liste des domaines uniques
domaines = merged_data["Domaine_emploi"].unique()

# Années futures pour les prédictions
future_years = np.arange(2022, 2051)

# Stocker les projections et les métriques
final_projections = []
model_metrics = []

# Ajuster et prédire pour chaque domaine
for domaine in domaines:
    # Filtrer les données pour un domaine spécifique
    domaine_data = merged_data[merged_data["Domaine_emploi"] == domaine]
    
    # Extraire les variables indépendantes (X) et dépendantes (y)
    X = domaine_data[["Année", "Valeur"]].dropna()
    y = domaine_data.loc[X.index, "Nombre d'emplois"]
    
    if X.empty or y.empty:
        continue  # Passer si pas assez de données pour ce domaine
    
    # Ajouter une constante pour le modèle
    X = sm.add_constant(X)
    
    # Calcul du VIF
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    ##print(f"\n--- VIF pour le domaine {domaine} ---\n{vif_data}")
    
    # Ajuster le modèle de régression
    model = sm.OLS(y, X).fit()
    
    # Imprimer les métriques du modèle
    ##print(f"\n--- Résultats pour le domaine {domaine} ---")
    ##print(model.summary())
    
    # Stocker les métriques
    model_metrics.append({
        "Domaine": domaine,
        "R2": model.rsquared,
        "R2 ajusté": model.rsquared_adj,
        "P-valeur (F-stat)": model.f_pvalue
    })
    
    # Préparer les prédicteurs pour les années futures
    future_data = pd.DataFrame({
        "Année": future_years,
        "Valeur": [X["Valeur"].mean()] * len(future_years)  # Moyenne des paramètres
    })
    future_X = sm.add_constant(future_data, has_constant="add")
    
    # Effectuer les prédictions
    future_predictions = model.predict(future_X)
    
    # Stocker les résultats dans la liste
    for year, pred in zip(future_years, future_predictions):
        final_projections.append({
            "Domaine": domaine,
            "Année": year,
            "Nombre d'emplois (prédit)": pred
        })

# Convertir les résultats en DataFrame
final_projections_df = pd.DataFrame(final_projections)

# Sauvegarder les prédictions dans un fichier CSV
final_projections_df.to_csv("projections_emplois_2050.csv", index=False)

# Convertir les métriques en DataFrame et les sauvegarder
model_metrics_df = pd.DataFrame(model_metrics)
model_metrics_df.to_csv("model_metrics.csv", index=False)

print("\nLes prédictions et les métriques des modèles ont été sauvegardées.")
