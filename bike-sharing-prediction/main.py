
# ============================================================
# LES RESEAUX DE NEURONES
# REGRESSION AVEC BIKE SHARING
# ============================================================


# ============================================================
# 1 - IMPORTATION ET CHARGEMENT DES DONNEES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Chargement du dataset
df = pd.read_csv("hour.csv")

print("===== 1 : Chargement des donnees =====")
print(df.head())


# ============================================================
# 2 - ANALYSE EXPLORATOIRE DES DONNEES
# ============================================================

print("\n===== 2 : Analyse exploratoire =====")

# Dimensions
print("Dimensions :", df.shape)

# Types des variables
print("\nInformations :")
df.info()

# Valeurs manquantes
print("\nValeurs manquantes :")
print(df.isna().sum())

# Statistiques descriptives
print("\nStatistiques descriptives :")
print(df.describe())


# ============================================================
# 3 - MATRICE DE CORRELATION
# ============================================================

print("\n===== 3 : Matrice de correlation =====")

plt.figure(figsize=(12, 7))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f"
)

plt.title("Matrice de correlation")
plt.tight_layout()
plt.show()


# ============================================================
# 4 - PREPARATION DES DONNEES
# ============================================================

print("\n===== 4 : Preparation des donnees =====")

# Supprimer les variables inutiles et celles qui provoquent
# une fuite de donnees (data leakage)
df_model = df.drop(
    columns=["instant", "dteday", "casual", "registered"]
)

# Séparer les caractéristiques et la cible
X = df_model.drop(columns="cnt")
y = df_model["cnt"]

print("Dimensions de X :", X.shape)
print("Dimensions de y :", y.shape)

print("\nCaracteristiques explicatives :")
print(X.columns.tolist())

print("\nVariable cible : cnt")


# ============================================================
# 5 - SEPARATION ET NORMALISATION DES DONNEES
# ============================================================

print("\n===== 5 : Separation et normalisation =====")

# Séparation : 80% apprentissage et 20% test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Extraire 20% de l'ensemble d'apprentissage pour validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42
)

print("X_train :", X_train.shape)
print("X_val :", X_val.shape)
print("X_test :", X_test.shape)

# Standardisation
scaler = StandardScaler()

# Ajuster uniquement sur les données d'apprentissage
X_train_scaled = scaler.fit_transform(X_train)

# Transformer validation et test
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 6 - CONSTRUCTION DU RESEAU DE NEURONES MLP
# ============================================================

print("\n===== 6 : Construction du modele =====")

# Fixer la graine aléatoire
tf.keras.utils.set_random_seed(42)

# Construction du modèle
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),

    Dense(64, activation="relu"),

    Dense(64, activation="relu"),

    Dense(32, activation="relu"),

    Dense(1)
])

# Compilation
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# Résumé du modèle
model.summary()


# ============================================================
# 7 - ENTRAINEMENT DU RESEAU
# ============================================================

print("\n===== 7 : Entrainement du modele =====")

# EarlyStopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

# Entraînement
history = model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=300,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

print("Epoques effectuees :", len(history.history["loss"]))


# ============================================================
# 8 - ANALYSE DE LA CONVERGENCE
# ============================================================

print("\n===== 8 : Courbes d'apprentissage =====")

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Apprentissage"
)

plt.plot(
    history.history["val_loss"],
    label="Validation"
)

plt.xlabel("Epoques")
plt.ylabel("MSE")
plt.title("Courbes d'apprentissage")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 9 - EVALUATION DU MODELE
# ============================================================

print("\n===== 9 : Evaluation du modele =====")

# Prédictions sur les données de test
y_pred = model.predict(
    X_test_scaled,
    verbose=0
).ravel()

# Calcul des métriques
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

# Affichage
print("MAE :", mae)
print("MSE :", mse)
print("R2 :", r2)


# ============================================================
# 10 - VISUALISATION DES PREDICTIONS
# ============================================================

print("\n===== 10 : Predictions et valeurs reelles =====")

plt.figure(figsize=(8, 5))

# Nuage de points
plt.scatter(
    y_test,
    y_pred,
    s=8,
    alpha=0.5
)

# Droite idéale y = x
low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())

plt.plot(
    [low, high],
    [low, high],
    "r--",
    label="Prediction ideale"
)

plt.xlabel("Valeurs reelles (cnt)")
plt.ylabel("Predictions")

plt.title("Predictions et valeurs reelles")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n===== FIN =====")
