import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. STYLE & CONFIG
st.set_page_config(page_title="Expert Crédit Rose", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #FFF0F5; }
    .decision-box { padding: 20px; border-radius: 15px; text-align: center; font-weight: bold; font-size: 22px; margin-bottom: 20px; }
    h1 { color: #C71585; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# 2. DONNÉES ET MODÈLE (Logique renforcée)
@st.cache_data
def load_and_train():
    np.random.seed(42)
    size = 3000
    data = pd.DataFrame({
        'person_age': np.random.randint(18, 80, size),
        'person_income': np.random.randint(10000, 150000, size),
        'person_emp_exp': np.random.randint(0, 30, size),
        'loan_amnt': np.random.randint(1000, 40000, size),
        'credit_score': np.random.randint(300, 850, size),
        'loan_status': np.random.choice([0, 1], size)
    })
    # On force le modèle à apprendre que Score Elevé + Dette Faible = Approuvé (0)
    data['loan_status'] = np.where((data['credit_score'] > 650) & (data['loan_amnt'] < data['person_income']*0.4), 0, 1)
    
    X = data.drop('loan_status', axis=1)
    y = data['loan_status']
    
    preprocessor = ColumnTransformer([('num', StandardScaler(), X.columns)])
    model = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])
    model.fit(X, y)
    return model

model = load_and_train()

# 3. INTERFACE
st.title("🌸 Assistant Crédit Intelligent")

with st.sidebar:
    st.header("💗 Paramètres")
    age = st.slider("Âge", 18, 90, 35)
    income = st.slider("Revenu Annuel ($)", 5000, 200000, 60000, step=1000)
    loan = st.slider("Montant du Prêt ($)", 500, 60000, 15000, step=500)
    score = st.slider("Credit Score", 300, 850, 720)
    exp = st.slider("Années d'Expérience", 0, 40, 10)

# 4. CALCULS
input_df = pd.DataFrame([[age, income, exp, loan, score]], 
                        columns=['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'credit_score'])

# Proba de risque (Classe 1)
risk_proba = model.predict_proba(input_df)[0][1]
ratio = loan / income

# 5. AFFICHAGE DES RÉSULTATS
st.write("---")
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Analyse du Risque")
    # JAUGE DE RISQUE
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_proba * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Indice de Danger", 'font': {'color': "#C71585"}},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#DB7093"},
            'steps' : [
                {'range': [0, 50], 'color': "#E6FFFA"},
                {'range': [50, 100], 'color': "#FFF5F5"}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
        }
    ))
    fig.update_layout(height=300, margin=dict(t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Verdict & Justificatif")
    
    # DECISION FINALE
    if risk_proba < 0.5:
        st.markdown('<div class="decision-box" style="background-color: #d4edda; color: #155724; border: 2px solid #28a745;">✅ DOSSIER APPROUVÉ</div>', unsafe_allow_html=True)
        st.success(f"Le profil est jugé sûr à {100-(risk_proba*100):.1f}%.")
    else:
        st.markdown('<div class="decision-box" style="background-color: #f8d7da; color: #721c24; border: 2px solid #dc3545;">❌ DOSSIER REJETÉ</div>', unsafe_allow_html=True)
        
        # VRAIE JUSTIFICATION
        st.write("**Points bloquants détectés :**")
        if ratio > 0.35:
            st.error(f"🔴 Ratio Dette/Revenu trop élevé ({ratio:.1%}). Le maximum conseillé est 35%.")
        if score < 600:
            st.error(f"🔴 Credit Score insuffisant ({score}). Un minimum de 600 est requis.")
        if income < 15000:
            st.error("🔴 Revenu trop faible pour garantir le remboursement.")
        if risk_proba >= 0.5 and ratio <= 0.35 and score >= 600:
            st.warning("⚠️ L'IA détecte une combinaison de facteurs à risque non spécifiée.")

st.write("---")
st.info("💡 Conseil : Pour obtenir une approbation, essayez d'augmenter le Credit Score ou de baisser le montant du prêt.")