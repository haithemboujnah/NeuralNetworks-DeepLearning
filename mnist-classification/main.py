
# ============================================================
# LES RESEAUX DE NEURONES
# CLASSIFICATION AVEC MNIST
# ============================================================


# ============================================================
# 1 - IMPORTATION ET PREPARATION DES DONNEES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Chargement du fichier CSV
data = pd.read_csv("mnist_784.csv")

print("===== 1 : Chargement des donnees =====")
print(data.head())

# Supprimer les lignes contenant des valeurs manquantes
n_before = len(data)
data = data.dropna()

print("Lignes incompletes supprimees :", n_before - len(data))

# Séparer les caractéristiques et les étiquettes
X = data.drop("class", axis=1).values
y = data["class"].astype(int).values

# Vérification de la structure
assert X.shape[1] == 784, "Chaque image doit contenir 784 pixels"
assert np.isfinite(X).all(), "Pixels non numeriques detectes"
assert X.min() >= 0 and X.max() <= 255, "Pixels hors de [0, 255]"

# Normalisation des pixels
X = X / 255.0


# ============================================================
# 2 - VERIFICATION DES DONNEES
# ============================================================

print("\n===== 2 : Verification des donnees =====")

print("Dimensions de X :", X.shape)
print("Min et Max :", X.min(), X.max())
print("Classes :", np.unique(y))

print("Nombre d'images :", X.shape[0])
print("Nombre de caracteristiques :", X.shape[1])


# ============================================================
# 3 - SEPARATION ET ENTRAINEMENT DU MLP
# ============================================================

print("\n===== 3 : Entrainement du MLP =====")

# Séparation des données : 80% apprentissage, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("X_train :", X_train.shape)
print("X_test :", X_test.shape)

# Construction du modèle MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(50, 20, 10),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=100,
    random_state=42
)

# Entraînement
mlp.fit(X_train, y_train)

print("Entrainement termine !")


# ============================================================
# 4 - EVALUATION DU MODELE
# ============================================================

print("\n===== 4 : Evaluation du modele =====")

# Prédictions
y_train_pred = mlp.predict(X_train)
y_test_pred = mlp.predict(X_test)

# Calcul des accuracies
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("Accuracy apprentissage :", train_acc)
print("Accuracy test :", test_acc)

# Rapport de classification
print("\nRapport de classification :")
print(classification_report(y_test, y_test_pred))


# ============================================================
# 5 - MATRICE DE CONFUSION
# ============================================================

print("\n===== 5 : Matrice de confusion =====")

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_test_pred,
    cmap="Blues",
    values_format="d"
)

plt.title("Matrice de confusion du MLP")
plt.tight_layout()
plt.show()


# ============================================================
# 6 - EVOLUTION DE LA FONCTION DE COUT (LOSS)
# ============================================================

print("\n===== 6 : Evolution de la Loss =====")

plt.figure(figsize=(7, 4))

epochs = np.arange(1, len(mlp.loss_curve_) + 1)

plt.plot(epochs, mlp.loss_curve_)

plt.xlabel("Epoques")
plt.ylabel("Fonction de cout (loss)")
plt.title("Evolution de la fonction de cout")

plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 7 - DIMENSIONS DES POIDS ET BIAIS
# ============================================================

print("\n===== 7 : Poids et biais =====")

# Afficher les dimensions des matrices de poids
for i, w in enumerate(mlp.coefs_):
    print(f"Poids {i + 1} :", w.shape)

# Afficher les dimensions des vecteurs de biais
for i, b in enumerate(mlp.intercepts_):
    print(f"Biais {i + 1} :", b.shape)

# Calculer le nombre total de paramètres
n_params = sum(w.size for w in mlp.coefs_) + \
           sum(b.size for b in mlp.intercepts_)

print("Nombre total de parametres :", n_params)

# Calcul théorique
print("\nCalcul theorique :")
print("Poids 1 :", 784 * 50)
print("Poids 2 :", 50 * 20)
print("Poids 3 :", 20 * 10)
print("Poids 4 :", 10 * 10)

print("Total poids :", 40500)
print("Total biais :", 50 + 20 + 10 + 10)
print("Total parametres :", 40590)


# ============================================================
# 8 - VALIDATION CROISEE (CROSS-VALIDATION)
# ============================================================

print("\n===== 8 : Validation croisee =====")

# Extraire un sous-ensemble stratifié de 6000 observations
X_cv, _, y_cv, _ = train_test_split(
    X_train,
    y_train,
    train_size=6000,
    stratify=y_train,
    random_state=42
)

# Validation croisée stratifiée à 3 partitions
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

# Modèle initial
scores = cross_val_score(
    MLPClassifier(
        hidden_layer_sizes=(50, 20, 10),
        learning_rate_init=0.001,
        max_iter=100,
        random_state=42
    ),
    X_cv,
    y_cv,
    cv=cv,
    scoring="accuracy",
    n_jobs=1
)

print("Accuracies :", scores)
print("Moyenne :", scores.mean())
print("Ecart-type :", scores.std())

# Sauvegarder la moyenne pour GridSearchCV
baseline_cv = scores.mean()


# ============================================================
# 9 - OPTIMISATION AVEC GRIDSEARCHCV
# ============================================================

print("\n===== 9 : Optimisation GridSearchCV =====")

# Définition des hyperparamètres
param_grid = {
    "hidden_layer_sizes": [
        (50,),
        (50, 20),
        (50, 20, 10)
    ],
    "activation": ["relu", "tanh"],
    "learning_rate_init": [0.001, 0.01]
}

# Création de GridSearchCV
search = GridSearchCV(
    estimator=MLPClassifier(
        max_iter=100,
        solver="adam",
        random_state=42
    ),
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=1,
    refit=True
)

# Recherche des meilleurs paramètres
search.fit(X_cv, y_cv)

print("Meilleurs parametres :", search.best_params_)
print("Meilleure accuracy CV :", search.best_score_)
print("Accuracy CV du MLP initial :", baseline_cv)

# Calcul du gain en points
print(
    "Gain CV (points) :",
    100 * (search.best_score_ - baseline_cv)
)


# ============================================================
# 10 - EVALUATION DU MODELE OPTIMISE
# ============================================================

print("\n===== 10 : Evaluation du modele optimise =====")

# Créer le meilleur modèle
best_mlp = MLPClassifier(
    **search.best_params_,
    max_iter=100,
    solver="adam",
    random_state=42
)

# Réentraînement sur toutes les données d'apprentissage
best_mlp.fit(X_train, y_train)

# Accuracy du modèle optimisé
best_test_acc = best_mlp.score(X_test, y_test)

print("Accuracy test (modele optimise) :", best_test_acc)

# Comparaison avec le modèle initial
print("Accuracy test (modele initial) :", test_acc)

print("\n===== FIN =====")
