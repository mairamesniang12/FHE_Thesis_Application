# 🔐 FHE Thesis Experimental Demonstrator

An interactive, browser-based demonstrator for the MSc thesis:

**"Efficient Privacy-Preserving ML Using Tree-Based and Hybrid Models Under FHE"**

**Mairame Samba NIANG**  
Supervised by **Dr. Célestin Wafo Soh**  
AIMS Sénégal  
2025–2026

---

## Overview

This application presents the experimental results obtained from the
thesis notebook on privacy-preserving machine learning using
**Fully Homomorphic Encryption (FHE)**.

The application provides an interactive interface for exploring:

- plaintext machine-learning baselines;
- FHE simulation results;
- real FHE execution results;
- plaintext vs FHE performance;
- FHE latency and confidence intervals;
- memory consumption and computational overhead;
- sensitivity to tree depth;
- sensitivity to the number of trees;
- sensitivity to quantization bits;
- client–server FHE deployment;
- neural preprocessing experiments;
- illustrative FHE + Differential Privacy experiments.

The application is designed as a **research demonstrator** for
presenting the experimental results of the thesis.

---

## 🔬 Experimental Scope

The main experiments cover five datasets:

- **WDBC** — Wisconsin Diagnostic Breast Cancer
- **Spambase**
- **Adult**
- **Pima Diabetes**
- **Heart Disease**

The main tree-based models are:

- Decision Tree
- Random Forest
- XGBoost

The main FHE experiments use **Concrete-ML 1.9.0**.

---

## Main FHE Experimental Protocol

The main FHE experiments recorded in the thesis notebook use:

- FHE quantization: `n_bits = 5`
- Decision Tree: `max_depth = 4`
- Random Forest: `n_estimators = 15`, `max_depth = 4`
- XGBoost: `n_estimators = 15`, `max_depth = 4`
- Calibration samples: `50`
- Real FHE evaluation samples: `30`
- FHE latency repetitions: `3`

The application reports:

- plaintext accuracy and F1-score;
- simulated FHE accuracy and F1-score;
- real FHE accuracy and F1-score;
- simulation/real prediction agreement;
- mean FHE latency;
- latency standard deviation;
- 95% confidence interval;
- peak RSS memory;
- accuracy difference between plaintext and FHE;
- computational overhead.

---

## 📊 Simulation vs Real FHE

The application distinguishes between two types of FHE evaluation.

### FHE simulation

The FHE-compatible model is evaluated in simulation on the complete
test set.

### Real FHE execution

The compiled FHE circuit is executed on a stratified subsample of
the test set.

This distinction is important because real FHE execution is
computationally expensive.

---

## 🖥️ Client–Server Demonstration

The application also includes a separate client–server experiment.

The deployment demonstration uses:

- WDBC dataset;
- Decision Tree;
- `max_depth = 5`;
- `n_bits = 6`.

The conceptual architecture is:

```text
                CLIENT
        ┌─────────────────────┐
        │ Quantization        │
        │ Encryption          │
        │                     │
        │ Private key         │
        └──────────┬──────────┘
                   │
                   │ Encrypted data
                   ▼
              NETWORK
                   │
                   ▼
        ┌─────────────────────┐
        │ FHE SERVER          │
        │                     │
        │ FHE inference       │
        │ on ciphertext       │
        └──────────┬──────────┘
                   │
                   │ Encrypted result
                   ▼
              NETWORK
                   │
                   ▼
                CLIENT
        ┌─────────────────────┐
        │ Decryption          │
        │ Dequantization      │
        │ Prediction          │
        └─────────────────────┘
