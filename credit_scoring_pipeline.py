import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# 1. Chargement des données
print("Chargement des données...")
try:
    df = pd.read_csv('loan_data.csv')
    print(f"Dataset chargé avec {df.shape[0]} lignes et {df.shape[1]} colonnes.")
except FileNotFoundError:
    print("Erreur : Le fichier 'loan_data.csv' est introuvable.")
    exit()

# 2. Prétraitement des données
print("\n--- Prétraitement ---")

# Nettoyage : Supprimer les aberrations (ex: person_age > 100)
print("Suppression des valeurs aberrantes (age > 100)...")
initial_rows = df.shape[0]
df = df[df['person_age'] <= 100]
print(f"Lignes supprimées : {initial_rows - df.shape[0]}")

# Séparation Features / Target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Identification des colonnes numériques et catégorielles
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

print(f"Variables numériques : {list(numerical_cols)}")
print(f"Variables catégorielles : {list(categorical_cols)}")

# Création des pipelines de transformation
# Numérique : Imputation par la médiane + Standard Scaling
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Catégoriel : Imputation par la valeur la plus fréquente + OneHotEncoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combiner les transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# 3. Entraînement et Comparaison
print("\n--- Entraînement et Comparaison ---")

# Division Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Taille Train : {X_train.shape}, Taille Test : {X_test.shape}")

# Définition des modèles
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = []

for name, model in models.items():
    print(f"\nEntraînement de {name}...")
    
    # Création du pipeline complet (Preprocess + Model)
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Évaluation
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1-Score": f1,
        "AUC-ROC": auc,
        "Pipeline": clf,
        "y_pred": y_pred
    })

# 4. Sélection du champion
print("\n--- Sélection du Champion ---")
results_df = pd.DataFrame(results).set_index("Model")
print(results_df[["Accuracy", "F1-Score", "AUC-ROC"]])

# Le critère est le F1-Score
best_model_name = results_df["F1-Score"].idxmax()
best_model_info = next(item for item in results if item["Model"] == best_model_name)
best_f1 = best_model_info["F1-Score"]

print(f"\n>> LE MEILLEUR MODÈLE EST : {best_model_name.upper()}")
print(f"RAISON : Il a le F1-Score le plus élevé ({best_f1:.4f}).")
print("Dans le risque de crédit, le F1-Score est crucial car il équilibre la précision (éviter les faux positifs) et le rappel (détecter tous les défauts potentiels), ce qui est plus informatif que l'accuracy sur des classes déséquilibrées.")

# 5. Matrice de confusion du meilleur modèle
print(f"\nAffichage de la matrice de confusion pour {best_model_name}...")
cm = confusion_matrix(y_test, best_model_info["y_pred"])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Matrice de Confusion - {best_model_name}')
plt.xlabel('Predit')
plt.ylabel('Réel')
plt.savefig('confusion_matrix.png')
print("Matrice de confusion sauvegardée sous 'confusion_matrix.png'")

print("\nScript terminé avec succès.")
