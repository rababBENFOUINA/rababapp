# -*- coding: utf-8 -*-
"""P2_ML.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZGtr0zwUXnOa9Jz25EpY3_irwy3-iVbj
"""

# Pandas est utilisé pour la manipulation et l'analyse de données tabulaires
import pandas as pd

# La fonction train_test_split de scikit-learn est utilisée pour diviser l'ensemble de données en ensembles d'entraînement et de test
from sklearn.model_selection import train_test_split

# RandomForestClassifier est un modèle d'ensemble d'arbres de décision, souvent utilisé pour des tâches de classification
from sklearn.ensemble import RandomForestClassifier

# accuracy_score est une fonction de scikit-learn qui mesure la précision du modèle en comparant les prédictions avec les valeurs réelles
from sklearn.metrics import accuracy_score ,classification_report

# LabelEncoder est utilisé pour convertir des valeurs catégorielles en nombres, nécessaire pour utiliser ces caractéristiques dans les modèles
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from joblib import dump

"""### code Python qui utilise la bibliothèque xml.etree.ElementTree pour lire des données depuis un fichier XML et la bibliothèque json pour lire des données depuis un fichier JSON, puis convertit ces données en format CSV à l'aide de la bibliothèque csv"""

# Importing library
import os


# La ligne suivante charge les données à partir d'un fichier CSV ("Book_Dataset.csv") dans un DataFrame appelé 'data'
data = pd.read_csv("C:\\Users\\HP\\Documents\\MIT\\ML\\projet\\projet_flask\\application\\routes\\GoodReadsAwards.csv")



# Calculer la médiane de la colonne 'c'
median_ratings = data['ratings'].median()
q1_ratings = data['ratings'].quantile(0.25)
q2_ratings = data['ratings'].quantile(0.75)
print(median_ratings)
print(q1_ratings)
print(q2_ratings)


"""Supprimer les lignes avec des valeurs manquantes :"""

data.dropna(inplace=True)
print(data.isnull().sum())

"""# Encodage des variables catégoriques :"""

label_encoder = LabelEncoder()
data['category_encoded'] = label_encoder.fit_transform(data['category'])


# Créer un DataFrame avec des catégories uniques et leurs encodages
unique_categories = data[['category', 'category_encoded']].drop_duplicates()

"""# Création de la variable cible (target) :"""

# Par exemple, si la note moyenne est supérieure à un seuil, définir comme recommandé
seuil_recommandation = 3.5
data['Recommande'] = ((data['avg_rating'] > seuil_recommandation ).astype(int) & (  data['ratings'] > q1_ratings).astype(int))
data.head()

df = data[['year', 'pages', 'category_encoded' , 'avg_rating' , 'Recommande']]

# Diviser les données en ensembles d'entraînement, de validation et de test
X_train, X_temp, y_train, y_temp = train_test_split(df.drop('Recommande', axis=1) , df['Recommande'], test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("ana hna wsalt")
# Définir les hyperparamètres à optimiser
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Créer le modèle de forêt aléatoire
model = RandomForestClassifier(random_state=42)

# Utiliser GridSearchCV pour optimiser les hyperparamètres
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Afficher les meilleurs hyperparamètres
print("Best Hyperparameters:", grid_search.best_params_)

# Utiliser le modèle avec les meilleurs hyperparamètres
best_model = grid_search.best_estimator_

# Entraîner le modèle sur l'ensemble d'entraînement
best_model.fit(X_train, y_train)

# Sauvegardez le modèle
dump(best_model, 'best_model.joblib')