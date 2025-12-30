import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. STYLE & CONFIGURATION
st.set_page_config(page_title="Expert Crédit Rose", layout="wide", page_icon="🌸")

# CSS Personnalisé pour un look plus moderne
st.markdown("""
    <style>
    .stApp { background-color: #FFF0F5; }
    .stButton>button { background-color: #C71585; color: white; border-radius: 10px; }
    h1, h2, h3 { color: #C71585; }
    .css-1d391kg { padding-top: 1rem; }
    .metric-card { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# 2. GÉNÉRATION DES DONNÉES ET MODÈLE
@st.cache_resource 
def load_and_train():
    np.random.seed(42)
    size = 5000 # Un peu plus de données pour la précision
    data = pd.DataFrame({
        'Âge': np.random.randint(18, 75, size),
        'Revenu': np.random.randint(15000, 150000, size),
        'Expérience': np.random.randint(0, 40, size),
        'Montant_Prêt': np.random.randint(2000, 50000, size),
        'Score_Crédit': np.random.randint(400, 850, size),
    })
    
    # Logique métier stricte pour l'entraînement (Target)
    # Refus si : Score < 620 OU (Dette > 40% Revenu)
    condition_refus = (data['Score_Crédit'] < 620) | (data['Montant_Prêt'] > data['Revenu'] * 0.40)
    data['Statut'] = np.where(condition_refus, 1, 0) # 1 = Risqué/Refus, 0 = OK
    
    X = data.drop('Statut', axis=1)
    y = data['Statut']
    
    # Pipeline
    preprocessor = ColumnTransformer([('num', StandardScaler(), X.columns)])
    # On limite la profondeur pour éviter le sur-apprentissage sur des données random
    model = Pipeline([
        ('pre', preprocessor), 
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])
    model.fit(X, y)
    
    return model, data

model, df_ref = load_and_train() # On récupère aussi les données pour les stats

# 3. INTERFACE UTILISATEUR
st.title("🌸 Expert Crédit Rose - IA Assistant")
st.markdown("---")

# Création de deux onglets pour alléger l'interface
tab1, tab2 = st.tabs(["📝 Simulation & Décision", "📊 Analyse & Comparaison"])

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4144/4144376.png", width=100)
    st.header("Profil du Demandeur")
    
    age = st.slider("Âge", 18, 80, 30)
    income = st.slider("Revenu Annuel (€)", 15000, 150000, 45000, step=1000)
    loan = st.slider("Montant demandé (€)", 2000, 50000, 10000, step=500)
    score = st.slider("Score de Crédit", 400, 850, 680)
    exp = st.slider("Années d'Expérience", 0, 40, 5)
    
    # Calculs intermédiaires
    ratio_dette = loan / income
    
    st.info(f"📊 Ratio Dette/Revenu actuel : **{ratio_dette:.1%}**")

# --- ONGLET 1 : DÉCISION ---
with tab1:
    # Préparation des données d'entrée
    input_df = pd.DataFrame([[age, income, exp, loan, score]], 
                            columns=['Âge', 'Revenu', 'Expérience', 'Montant_Prêt', 'Score_Crédit'])

    # Prédiction
    risk_proba = model.predict_proba(input_df)[0][1] # Probabilité d'être classe 1 (Refus)
    
    col_gauche, col_droite = st.columns([1, 2])
    
    with col_gauche:
        st.subheader("Verdict de l'IA")
        # Jauge améliorée
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_proba * 100,
            number = {'suffix': "%", 'font': {'color': "#C71585"}},
            title = {'text': "Probabilité de Risque"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#C71585" if risk_proba > 0.5 else "#28a745"},
                'steps' : [
                    {'range': [0, 50], 'color': "white"},
                    {'range': [50, 100], 'color': "#ffe6e6"}],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=30, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_droite:
        st.subheader("Analyse du Dossier")
        
        # Affichage avec st.metric pour un look pro
        m1, m2, m3 = st.columns(3)
        m1.metric("Montant", f"{loan} €")
        m2.metric("Revenus", f"{income} €")
        m3.metric("Score", f"{score}", delta=score-620, delta_color="normal")

        st.divider()

        if risk_proba < 0.5:
            st.success("### ✅ DOSSIER APPROUVÉ")
            st.write("Le profil respecte les critères de sécurité financière établis par l'algorithme.")
        else:
            st.error("### ❌ DOSSIER REJETÉ")
            st.write("Ce dossier présente un risque trop élevé.")
            
            # Explications dynamiques (plus intelligentes)
            with st.expander("🔍 Comprendre le refus (Détails)"):
                if ratio_dette > 0.40:
                    st.write(f"🔴 **Surendettement :** Le prêt représente {ratio_dette:.0%} du revenu (Max conseillé : 40%).")
                if score < 620:
                    st.write(f"🔴 **Historique Bancaire :** Score de {score} trop faible (Min requis : 620).")
                if income < 20000:
                    st.write("⚠️ **Revenus :** Les revenus sont dans la fourchette basse de nos clients acceptés.")

# --- ONGLET 2 : ANALYSE AVANCÉE ---
with tab2:
    st.subheader("📈 Importance des critères pour l'IA")
    st.write("Qu'est-ce qui influence le plus la décision de l'algorithme ?")
    
    # Extraction de l'importance des features
    importances = model.named_steps['clf'].feature_importances_
    feature_names = input_df.columns
    df_imp = pd.DataFrame({'Critère': feature_names, 'Importance': importances}).sort_values('Importance', ascending=True)
    
    fig_imp = px.bar(df_imp, x='Importance', y='Critère', orientation='h', 
                     color='Importance', color_continuous_scale='RdPu') # Echelle Rose-Violet
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.divider()
    
    st.subheader("📍 Votre position par rapport aux autres clients")
    # Comparaison Score Crédit
    fig_hist = px.histogram(df_ref, x="Score_Crédit", nbins=50, color_discrete_sequence=['#FFB6C1'], title="Distribution des Scores Crédit")
    # Ajouter une ligne verticale pour le client actuel
    fig_hist.add_vline(x=score, line_width=3, line_dash="dash", line_color="#C71585", annotation_text="Vous êtes ici", annotation_position="top left")
    st.plotly_chart(fig_hist, use_container_width=True)