import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# -----------------------------------------------------------------------------
# 1. CONFIGURATION DE LA PAGE & STYLE (Design Sophistiqué)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Conseiller Crédit IA", page_icon="🏦", layout="wide")

# CSS Personnalisé pour un look moderne et "bancaire"
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    div.stButton > button:first-child {
        background-color: #2e86de; color: white; border-radius: 10px; 
        padding: 10px 24px; border: none; font-weight: bold;
    }
    div.stButton > button:hover { background-color: #54a0ff; border: none; }
    .metric-card {
        background-color: white; padding: 20px; border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;
    }
    .header-style { font-size: 24px; font-weight: 700; color: #34495e; }
    .success-box { padding: 20px; background-color: #d4edda; color: #155724; border-radius: 10px; border: 1px solid #c3e6cb; }
    .error-box { padding: 20px; background-color: #f8d7da; color: #721c24; border-radius: 10px; border: 1px solid #f5c6cb; }
</style>
""", unsafe_allow_html=True)

st.title("🏦 Assistant Bancaire Intelligent")
st.markdown("### Système d'Aide à la Décision Crédit (Multi-Modèles)")

# -----------------------------------------------------------------------------
# 2. CHARGEMENT ET PRÉPARATION DES DONNÉES
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Simulation de données si pas de CSV (pour que l'app marche tout de suite)
    np.random.seed(42)
    size = 2000 # Réduit pour la démo, 45000 en prod
    data = pd.DataFrame({
        'person_age': np.random.randint(20, 110, size), # Avec aberrations > 100
        'person_income': np.random.randint(20000, 150000, size),
        'person_home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE', 'OTHER'], size),
        'person_emp_length': np.random.randint(0, 40, size),
        'loan_intent': np.random.choice(['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL'], size),
        'loan_grade': np.random.choice(['A', 'B', 'C', 'D', 'E'], size),
        'loan_amnt': np.random.randint(1000, 35000, size),
        'loan_int_rate': np.round(np.random.uniform(5, 20, size), 2),
        'loan_percent_income': np.random.uniform(0.05, 0.6, size),
        'cb_person_default_on_file': np.random.choice(['Y', 'N'], size),
        'loan_status': np.random.choice([0, 1], size, p=[0.8, 0.2]) # 0=Rejet, 1=Approuvé (Déséquilibré)
    })
    return data

df = load_data()

# --- Nettoyage ---
# Suppression des aberrations (Age > 100) comme demandé
df_clean = df[df['person_age'] <= 100].copy()

# Séparation Features / Target
X = df_clean.drop('loan_status', axis=1)
y = df_clean['loan_status']

# Définition des colonnes
num_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income']
cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Pipeline de Preprocessing
# Imputation + Scaling pour numérique / Imputation + OneHot pour catégoriel
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------------------------------------------
# 3. ENTRAÎNEMENT & SÉLECTION DU CHAMPION
# -----------------------------------------------------------------------------
@st.cache_resource
def train_models(_X_train, _y_train, _X_test, _y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    results = {}
    best_model_name = ""
    best_f1 = -1
    best_pipeline = None

    for name, model in models.items():
        # Création du pipeline complet
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        clf.fit(_X_train, _y_train)
        y_pred = clf.predict(_X_test)
        
        # Métriques
        f1 = f1_score(_y_test, y_pred)
        acc = accuracy_score(_y_test, y_pred)
        try:
            auc = roc_auc_score(_y_test, clf.predict_proba(_X_test)[:, 1])
        except:
            auc = 0.5
            
        results[name] = {"F1": f1, "Accuracy": acc, "AUC": auc}
        
        # Sélection du champion sur F1 Score
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = clf
            
    return results, best_model_name, best_pipeline

results, champion_name, champion_model = train_models(X_train, y_train, X_test, y_test)

# -----------------------------------------------------------------------------
# 4. INTERFACE UTILISATEUR (SIDEBAR)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4149/4149665.png", width=80)
    st.header("📝 Dossier Client")
    
    input_data = {}
    input_data['person_age'] = st.slider("Âge", 18, 100, 30)
    input_data['person_income'] = st.number_input("Revenu Annuel (€)", 10000, 200000, 50000)
    input_data['person_home_ownership'] = st.selectbox("Propriété", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    input_data['person_emp_length'] = st.slider("Années d'emploi", 0, 50, 5)
    input_data['loan_intent'] = st.selectbox("Motif du prêt", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL'])
    input_data['loan_grade'] = st.selectbox("Grade (Score externe)", ['A', 'B', 'C', 'D', 'E'])
    input_data['loan_amnt'] = st.number_input("Montant demandé (€)", 1000, 50000, 10000)
    input_data['loan_int_rate'] = st.slider("Taux d'intérêt (%)", 5.0, 25.0, 10.0)
    input_data['cb_person_default_on_file'] = st.radio("Défaut historique ?", ['N', 'Y'])
    
    # Calcul automatique
    input_data['loan_percent_income'] = input_data['loan_amnt'] / input_data['person_income']
    st.caption(f"Ratio Dette/Revenu calculé : {input_data['loan_percent_income']:.2%}")

    predict_btn = st.button("🔍 Analyser le Dossier")

# -----------------------------------------------------------------------------
# 5. AFFICHAGE DES RÉSULTATS (MAIN)
# -----------------------------------------------------------------------------

# Section Performance Modèles
with st.expander("📊 Voir les performances techniques des modèles (Backend ML)"):
    col1, col2, col3 = st.columns(3)
    for model_name, metrics in results.items():
        is_champion = "👑 " if model_name == champion_name else ""
        col1.write(f"**{is_champion}{model_name}**")
        col2.progress(metrics['F1'])
        col3.caption(f"F1: {metrics['F1']:.2f} | AUC: {metrics['AUC']:.2f}")
    st.info(f"Le modèle Champion sélectionné automatiquement est **{champion_name}** car il a le meilleur F1-Score (équilibre Précision/Rappel).")

if predict_btn:
    # Création du DataFrame pour la prédiction
    input_df = pd.DataFrame([input_data])
    
    # Prédiction
    probabilite_risque = champion_model.predict_proba(input_df)[0][1] # Proba de la classe 1 (Défaut/Risque élevé dans ce dataset simulé)
    prediction = champion_model.predict(input_df)[0] # 0 ou 1
    
    # Interprétation métier : Dans ce dataset simulé, 1 = Risque/Refus, 0 = OK (à adapter selon vos vraies labels)
    # INVERSION POUR LA LOGIQUE BANCAIRE USUELLE : 
    # Souvent Target 1 = Défaut (Mauvais). Donc Score élevé = Danger.
    
    score_credit = int(probabilite_risque * 100)
    
    st.divider()
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        # Jauge Plotly
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score_credit,
            title = {'text': "Probabilité de Défaut"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgreen"},
                    {'range': [40, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "salmon"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': score_credit}}))
        fig.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Décision Recommandée")
        
        # Logique de décision (Seuil à 50% par défaut)
        if score_credit < 50:
            st.markdown(f"""
            <div class="success-box">
                <h2>✅ CRÉDIT APPROUVÉ</h2>
                <p>Le risque de défaut est estimé à seulement {score_credit}%. Le profil du client est solide selon le modèle {champion_name}.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="error-box">
                <h2>⚠️ CRÉDIT REJETÉ (Risque Élevé)</h2>
                <p>Attention : La probabilité de défaut est de {score_credit}%. Ce niveau dépasse le seuil de tolérance de la banque.</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Feature Importance (Explainability)
        st.write("#### 🔎 Pourquoi cette décision ? (Facteurs Clés)")
        
        # Extraction de l'importance des features (Approximation pour Pipeline)
        # On récupère le modèle final
        model_step = champion_model.named_steps['classifier']
        
        if hasattr(model_step, 'feature_importances_'):
            importances = model_step.feature_importances_
            # On doit récupérer les noms des features après OneHot (c'est un peu technique avec Pipeline)
            # Pour simplifier l'affichage visuel sans casser le code :
            # On affiche les importances numériques brutes mappées aux colonnes numériques principales
            # (Note: Une solution parfaite nécessiterait get_feature_names_out, complexe à coder en un script simple)
            
            feat_imp = pd.DataFrame({
                'Feature': num_cols, # Simplification pour l'exemple visuel
                'Importance': importances[:len(num_cols)] # On prend les premières correspondantes
            }).sort_values(by='Importance', ascending=False)
            
            fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', 
                             title="Impact des variables numériques", color='Importance', color_continuous_scale='Blues')
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Les coefficients détaillés ne sont pas disponibles pour ce modèle spécifique sous cette forme visuelle.")