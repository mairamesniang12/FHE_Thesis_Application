# FHE Thesis Application 

Application Streamlit construite à partir du notebook `source_notebook.ipynb`.

## Lancer

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Contenu

- Dashboard des 5 datasets et 3 modèles
- Plaintext vs FHE
- Analyse de sensibilité : profondeur, nombre d'arbres, quantification
- Déploiement client–serveur FHE
- Neural preprocessing + baseline MLP
- Simulation FHE + DP
- Méthodologie et provenance

Les résultats sont stockés dans `data/`. L'application ne ré-exécute pas les expériences FHE coûteuses.
