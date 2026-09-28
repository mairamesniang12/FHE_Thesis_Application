
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="FHE Thesis Demonstrator", page_icon="🔐", layout="wide")

ROOT=Path(__file__).parent
DATA=ROOT/"data"

@st.cache_data
def load(name): return pd.read_csv(DATA/name)

master=load("master_results.csv")
depth=load("depth_results.csv")
trees=load("ntrees_results.csv")
bits=load("bits_results.csv")
dp=load("dp_simulation_results.csv")
neural=load("neural_preprocessing_results.csv")
mlp=load("mlp_fhe_results.csv")
deploy=load("client_server_results.csv")

st.sidebar.title("🔐 FHE Thesis")
page=st.sidebar.radio("Navigation",[
    "🏠 Accueil","📊 Dashboard","📈 Plaintext vs FHE",
    "🔬 Sensibilité","🖥️ Client–Serveur","🧠 Neural preprocessing",
    "🔒 FHE + DP","📚 Méthodologie"
])
st.sidebar.caption("Résultats extraits du notebook fourni")
st.sidebar.caption("Concrete-ML 1.9.0 • n_bits=5 pour l'expérience principale")

def pct(x): return f"{x*100:.2f}%"

if page=="🏠 Accueil":
    st.title("🔐 FHE Privacy-Preserving Machine Learning")
    st.subheader("Démonstrateur interactif de la thèse")
    st.write("Cette application présente les résultats expérimentaux enregistrés dans le notebook de référence, avec une séparation explicite entre simulation FHE, exécution FHE réelle et simulation Monte Carlo DP.")
    a,b,c,d=st.columns(4)
    a.metric("Datasets","5")
    b.metric("Modèles","3")
    c.metric("Expériences FHE","15")
    d.metric("Répétitions FHE","3")
    st.divider()
    st.subheader("Architecture")
    st.code("""Données utilisateur
       ↓
Client FHE — quantification + chiffrement
       ↓
Réseau
       ↓
Serveur FHE — inférence sur ciphertext
       ↓
Réseau
       ↓
Client — déchiffrement
       ↓
Prédiction""")
    st.info("Les métriques affichées proviennent du notebook fourni. L'application ne ré-exécute pas automatiquement les expériences coûteuses.")

elif page=="📊 Dashboard":
    st.title("📊 Dashboard expérimental")
    ds=st.multiselect("Datasets",master.dataset.unique(),default=list(master.dataset.unique()))
    models=st.multiselect("Modèles",master.model.unique(),default=list(master.model.unique()))
    df=master[master.dataset.isin(ds)&master.model.isin(models)].copy()
    c1,c2,c3=st.columns(3)
    c1.metric("Configurations",len(df))
    c2.metric("Acc. FHE moyenne",pct(df.acc_fhe_sim.mean()))
    c3.metric("Acc. plaintext moyenne",pct(df.acc_plain.mean()))
    show=df[["dataset","model","acc_plain","acc_fhe_sim","delta_acc","acc_fhe_real","agreement","latency_s","ci_low_s","ci_high_s","memory_mb","overhead_ratio"]].copy()
    st.dataframe(show.style.format({
        "acc_plain":"{:.4f}","acc_fhe_sim":"{:.4f}","delta_acc":"{:+.4f}",
        "acc_fhe_real":"{:.4f}","agreement":"{:.2f}","latency_s":"{:.4f}",
        "ci_low_s":"{:.4f}","ci_high_s":"{:.4f}","memory_mb":"{:.1f}","overhead_ratio":"{:.0f}x"
    }),use_container_width=True,hide_index=True)
    st.subheader("Latence FHE par dataset et modèle")
    chart=df.pivot(index="dataset",columns="model",values="latency_s")
    st.bar_chart(chart)

elif page=="📈 Plaintext vs FHE":
    st.title("📈 Comparaison Plaintext / FHE")
    metric=st.selectbox("Métrique",["Accuracy","F1-score"])
    if metric=="Accuracy":
        tmp=master[["dataset","model","acc_plain","acc_fhe_sim"]].melt(["dataset","model"],var_name="mode",value_name="value")
    else:
        tmp=master[["dataset","model","f1_plain","f1_fhe_sim"]].melt(["dataset","model"],var_name="mode",value_name="value")
    tmp["configuration"]=tmp.dataset+" — "+tmp.model
    st.bar_chart(tmp.pivot(index="configuration",columns="mode",values="value"))
    st.subheader("Latence FHE et intervalle de confiance")
    st.dataframe(master[["dataset","model","latency_s","latency_std_s","ci_low_s","ci_high_s"]].style.format("{:.4f}"),use_container_width=True,hide_index=True)
    st.caption("L'accuracy FHE simulée est calculée sur l'ensemble de test; l'accuracy FHE réelle est rapportée sur un sous-échantillon stratifié de 30 observations.")

elif page=="🔬 Sensibilité":
    st.title("🔬 Analyse de sensibilité")
    section=st.radio("Analyse",["Profondeur d'arbre","Nombre d'arbres","Quantification"],horizontal=True)
    if section=="Profondeur d'arbre":
        d=depth[depth.dataset.isin(st.multiselect("Datasets",depth.dataset.unique(),default=list(depth.dataset.unique()),key="depds"))]
        st.line_chart(d.pivot(index="max_depth",columns="dataset",values="latency_s"))
        st.subheader("Accuracy FHE simulée")
        st.line_chart(d.pivot(index="max_depth",columns="dataset",values="accuracy"))
        st.dataframe(d,use_container_width=True,hide_index=True)
    elif section=="Nombre d'arbres":
        st.line_chart(trees.pivot(index="n_estimators",columns="dataset",values="latency_s"))
        st.dataframe(trees,use_container_width=True,hide_index=True)
    else:
        st.line_chart(bits.pivot(index="n_bits",columns="dataset",values="latency_s"))
        st.subheader("Accuracy FHE simulée")
        st.line_chart(bits.pivot(index="n_bits",columns="dataset",values="accuracy"))
        st.dataframe(bits,use_container_width=True,hide_index=True)

elif page=="🖥️ Client–Serveur":
    st.title("🖥️ Déploiement Client–Serveur FHE")
    st.write("Expérience WDBC + Decision Tree, max_depth=5, n_bits=6, basée sur FHEModelDev/FHEModelClient/FHEModelServer.")
    a,b,c,d=st.columns(4)
    a.metric("Chiffrement","12.6 ms")
    b.metric("Serveur FHE","3.847 s")
    c.metric("Déchiffrement","3.5 ms")
    d.metric("End-to-end","7.514 s")
    st.subheader("Flux")
    st.code("""CLIENT
  quantize_encrypt_serialize()
       │  request = 984 bytes
       ▼
SERVER
  server.run(ciphertext, evaluation_keys)
       │  response = 33056 bytes
       ▼
CLIENT
  deserialize_decrypt_dequantize()
""")
    st.dataframe(deploy,use_container_width=True,hide_index=True)
    st.warning("Les temps réseau et end-to-end ci-dessus proviennent du démonstrateur à deux processus. Le calcul 'Total no network' additionne chiffrement + inférence serveur + déchiffrement.")

elif page=="🧠 Neural preprocessing":
    st.title("🧠 Neural preprocessing + Tree")
    st.write("Autoencoder : 32-ReLU → 16-ReLU → bottleneck linéaire de dimension 8 → décodeur symétrique.")
    st.dataframe(neural,use_container_width=True,hide_index=True)
    st.subheader("Latence brute vs bottleneck")
    st.bar_chart(neural.set_index("dataset")[["latency_raw_s","latency_bottleneck_s"]])
    st.subheader("Baseline FHE-MLP")
    st.dataframe(mlp,use_container_width=True,hide_index=True)
    st.warning("Le notebook n'a pas finalisé l'assemblage de la comparaison H1 dans df_h1 : le test MLP réel a été exécuté mais l'agrégation a échoué avec NameError. Les résultats MLP sont donc affichés séparément, sans conclure une comparaison H1 complète.")

elif page=="🔒 FHE + DP":
    st.title("🔒 FHE + Differential Privacy")
    st.write("Simulation Monte Carlo illustrative d'un mécanisme de perturbation des sorties; 20 répétitions par configuration, δ=10⁻⁵.")
    st.dataframe(dp,use_container_width=True,hide_index=True)
    sel=st.selectbox("Dataset",dp.dataset.unique())
    d=dp[dp.dataset==sel].copy()
    st.line_chart(d.pivot(index="epsilon",columns="model",values="acc_dp_mean"))
    st.warning("Le notebook demande explicitement de présenter cette section comme une simulation illustrative, et non comme une preuve formelle de garantie (ε,δ)-DP du modèle FHE. Δf=1 est une hypothèse simplificatrice et le bruit est ajouté aux probabilités du modèle plaintext.")

else:
    st.title("📚 Méthodologie")
    st.markdown("""
**Datasets :** WDBC, Spambase, Adult, Pima Diabetes et Heart Disease.

**Split :** 80/20, `random_state=42`, stratification.

**Modèles plaintext :**
- Decision Tree : `max_depth=5`
- Random Forest : `n_estimators=50`, `max_depth=5`
- XGBoost : `n_estimators=50`, `max_depth=5`

**Expériences FHE principales :**
- Concrete-ML 1.9.0
- `n_bits=5`
- 3 répétitions de latence
- 30 observations pour l'exécution FHE réelle
- accuracy/F1 de simulation sur l'ensemble de test
- accuracy/F1 réels sur le sous-échantillon
- accord simulation/réel
- IC 95 % de la latence
- mémoire RSS maximale

**Analyse de sensibilité :**
- profondeur : 3, 5, 7, 10
- nombre d'arbres : 10, 50, 100
- bits : 2, 4, 6, 8

**Client–serveur :** WDBC + DT, `max_depth=5`, `n_bits=6`.

**Provenance :** toutes les données de cette interface sont fournies dans `data/` et le notebook source est inclus à la racine du projet.
""")
    st.subheader("Fichiers de résultats")
    st.write([p.name for p in DATA.iterdir()])
