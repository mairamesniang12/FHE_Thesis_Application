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
presenting and exploring the experimental results of the thesis.

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

The application therefore presents both results separately rather
than treating simulated and real FHE execution as identical
experiments.

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
```

This experiment is presented separately from the main 15
FHE configurations.

---

## 📁 Project Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
│
└── data/
    ├── master_results.csv
    ├── depth_results.csv
    ├── ntrees_results.csv
    ├── bits_results.csv
    ├── dp_simulation_results.csv
    ├── neural_preprocessing_results.csv
    ├── mlp_fhe_results.csv
    └── client_server_results.csv
```

The `data/` directory contains the experimental results exported
from the thesis notebook.

The Streamlit application reads these CSV files and presents the
results interactively.

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone <https://github.com/mairamesniang12/FHE_Thesis_Application/tree/master>
```

Then enter the project directory:

```bash
cd <FHE_Thesis_Application>
```

### 2. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit application

```bash
streamlit run app.py
```

After launching the application, Streamlit will provide a local
address such as:

```text
http://localhost:8501
```

Open this address in your web browser.

### Requirements

The application requires:

```text
streamlit>=1.35
pandas>=2.2
numpy>=1.26
```

The Streamlit interface does not re-run the computationally
expensive FHE experiments.

The experimental results are loaded from the CSV files stored in
the `data/` directory.

---

## ☁️ Deployment with Streamlit Community Cloud

The application can be deployed online using
**Streamlit Community Cloud**.

### Steps

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new application.
4. Select this GitHub repository.
5. Select `app.py` as the main application file.
6. Deploy the application.

The repository must contain the `data/` directory and all required
CSV files.

Once deployed, Streamlit provides a public URL that can be shared
with supervisors, collaborators, or examiners.

---

## 📊 Application Sections

The application contains the following sections.

### 🏠 Home

Provides an overview of the thesis, datasets, models, FHE
experiments, and the client–server architecture.

### 📊 Experimental Dashboard

Provides an interactive overview of the main FHE experiments,
including:

- accuracy;
- F1-score;
- FHE simulation results;
- real FHE results;
- simulation/real agreement;
- latency;
- confidence intervals;
- memory consumption;
- computational overhead.

Users can filter the results by dataset and model.

### 📈 Plaintext vs FHE

Allows comparison between plaintext and FHE performance.

The application distinguishes between:

- plaintext inference;
- simulated FHE inference;
- real FHE execution.

The page also presents FHE latency together with the recorded
95% confidence intervals.

### 🔬 Sensitivity Analysis

Provides three sensitivity analyses:

- tree depth;
- number of trees;
- quantization bits.

The corresponding latency and predictive-performance results are
displayed interactively.

### 🖥️ Client–Server Deployment

Presents the separate WDBC client–server FHE demonstration,
including encryption, server inference, decryption, and
end-to-end timing.

### 🧠 Neural Preprocessing

Displays the neural preprocessing and FHE-MLP experimental
results recorded in the thesis notebook.

### 🔒 FHE + Differential Privacy

Displays the illustrative Monte Carlo Differential Privacy
simulation.

This section is presented as an illustrative simulation and not
as a formal proof of an `(ε, δ)`-Differential Privacy guarantee.

### 📚 Methodology

Summarizes the datasets, models, FHE configuration, sensitivity
experiments, and experimental methodology used in the thesis.

---

## 🔬 Sensitivity Analysis

The application provides three sensitivity analyses.

### Tree Depth

The notebook evaluates different tree-depth configurations:

```text
3
5
7
10
```

### Number of Trees

The notebook evaluates:

```text
10
50
100
```

### Quantization Bits

The notebook evaluates:

```text
2
4
6
8
```

The application displays the corresponding experimental results
for latency and predictive performance.

---

## 🧠 Neural Preprocessing

The application includes the neural preprocessing experiments
recorded in the thesis notebook.

The neural preprocessing experiment investigates a compressed
representation before the downstream machine-learning evaluation.

The application also displays the recorded FHE-MLP results.

These results are presented separately from the main tree-based
FHE experiments.

---

## 🔒 FHE + Differential Privacy

The application includes an illustrative Monte Carlo simulation
of output perturbation.

This section is explicitly presented as a **simulation**.

It should not be interpreted as a formal proof of an
`(ε, δ)`-Differential Privacy guarantee for the FHE model.

The purpose of this section is to provide an experimental
illustration of the interaction between privacy-preserving
mechanisms and predictive performance.

---

## ⚠️ Scope and Limitations

This application is a **research demonstrator** accompanying the
MSc thesis.

It is important to distinguish the Streamlit interface from the
experimental notebook.

The computationally expensive FHE experiments were executed in
the thesis notebook. Their results were exported to CSV files and
are displayed by this application.

Therefore:

- the Streamlit application does not automatically recompile the
  FHE models;
- the Streamlit application does not repeat the complete FHE
  benchmark;
- the displayed experimental values come from the recorded
  notebook results;
- the client–server page represents a separate deployment
  demonstration;
- the thesis remains the primary source for the complete
  experimental methodology and statistical analysis.

The application is intended primarily for **interactive
presentation, visualization, and exploration of the experimental
results**.

---

## 📚 Data Provenance

The main experimental results are stored in:

```text
data/master_results.csv
```

Additional experiments are stored in:

```text
data/depth_results.csv
data/ntrees_results.csv
data/bits_results.csv
data/dp_simulation_results.csv
data/neural_preprocessing_results.csv
data/mlp_fhe_results.csv
data/client_server_results.csv
```

These files contain results generated from the thesis notebook.

The application reads these files at runtime and does not
reconstruct the experiments from scratch.

---

## 🔁 Reproducibility

The Streamlit application is intended for **interactive
presentation and exploration** of the experimental results.

The original thesis notebook contains the experimental code used
to generate the FHE results.

The CSV files included with this application allow the results to
be explored without repeating the computationally expensive FHE
experiments.

The application therefore separates:

```text
Experimental computation
        │
        ▼
Thesis notebook
        │
        ▼
CSV result files
        │
        ▼
Streamlit application
        │
        ▼
Interactive visualization
```

This separation makes it possible to present the experimental
results without requiring the supervisor to rerun the complete
FHE pipeline.

---

## 📦 Software Environment

The main experimental environment used in the thesis includes:

```text
Python: 3.12.13
NumPy: 1.26.4
Pandas: 2.2.2
Scikit-learn: 1.5.0
XGBoost: 1.6.2
Concrete-ML: 1.9.0
```

The Streamlit visualization layer itself only requires the
packages listed in `requirements.txt`.

---

## 📜 License

The demonstrator code is released under the **MIT License**.

See the `LICENSE` file for the complete license text.

Concrete-ML is a separate software project and is distributed
under its own license and terms.

---

## 🙏 Acknowledgements

This work uses **Concrete-ML**, developed by Zama, for
Fully Homomorphic Encryption-compatible machine learning.

This demonstrator was developed as part of the MSc research
project at **AIMS Sénégal**.

---

## 👤 Author

**Mairame Samba NIANG**

MSc Research Project  
AIMS Sénégal  
2025–2026

**Supervisor:** Dr. Célestin Wafo Soh
