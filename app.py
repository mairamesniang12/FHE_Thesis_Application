import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FHE Thesis Demonstrator",
    page_icon="🔐",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).parent
DATA = ROOT / "data"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_csv(filename):
    path = DATA / filename

    if not path.exists():
        st.error(f"Required data file not found: {path}")
        st.stop()

    return pd.read_csv(path)


# Main experimental results
master = load_csv("master_results.csv")

# Sensitivity experiments
depth = load_csv("depth_results.csv")
trees = load_csv("ntrees_results.csv")
bits = load_csv("bits_results.csv")

# Additional experiments
dp = load_csv("dp_simulation_results.csv")
neural = load_csv("neural_preprocessing_results.csv")
mlp = load_csv("mlp_fhe_results.csv")
deploy = load_csv("client_server_results.csv")


# ============================================================
# BASIC VALIDATION
# ============================================================

required_master_columns = [
    "dataset",
    "model",
    "acc_plain",
    "f1_plain",
    "acc_fhe_simulate_full",
    "f1_fhe_simulate_full",
    "acc_fhe_real_subsample",
    "f1_fhe_real_subsample",
    "agreement_simulate_vs_real",
    "latency_fhe_mean_s",
    "latency_fhe_std_s",
    "latency_ci95_low_s",
    "latency_ci95_high_s",
    "memory_peak_rss_mb",
    "delta_acc_fhe_minus_plain",
    "overhead_ratio"
]

missing_columns = [
    col for col in required_master_columns
    if col not in master.columns
]

if missing_columns:
    st.error(
        "The following columns are missing from master_results.csv:\n\n"
        + "\n".join(f"- {c}" for c in missing_columns)
    )
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔐 FHE Thesis")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Experimental Dashboard",
        "📈 Plaintext vs FHE",
        "🔬 Sensitivity Analysis",
        "🖥️ Client–Server Deployment",
        "🧠 Neural Preprocessing",
        "🔒 FHE + Differential Privacy",
        "📚 Methodology"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Experimental results extracted from the corrected thesis notebook."
)

st.sidebar.caption(
    "Concrete-ML 1.9.0 • Main FHE experiment: n_bits=5"
)


# ============================================================
# HELPERS
# ============================================================

def pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.2f}%"


def safe_mean(series):
    if series.empty:
        return np.nan
    return series.mean()


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title("🔐 Privacy-Preserving Machine Learning with FHE")

    st.subheader(
        "Interactive thesis experimental demonstrator"
    )

    st.write(
        """
        This application presents the experimental results recorded in
        the thesis notebook. It provides an interactive view of the
        plaintext baselines, Fully Homomorphic Encryption (FHE)
        experiments, sensitivity analyses, client–server deployment,
        neural preprocessing, and illustrative Differential Privacy
        experiments.
        """
    )

    st.info(
        """
        The displayed metrics come from the thesis notebook.
        The application does not automatically re-run the
        computationally expensive FHE experiments.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # OVERVIEW METRICS
    # --------------------------------------------------------

    datasets_count = master["dataset"].nunique()
    models_count = master["model"].nunique()
    configurations_count = len(master)

    a, b, c, d = st.columns(4)

    a.metric(
        "Datasets",
        datasets_count
    )

    b.metric(
        "Models",
        models_count
    )

    c.metric(
        "Main FHE Configurations",
        configurations_count
    )

    d.metric(
        "FHE Quantization",
        "5 bits"
    )

    st.divider()

    # --------------------------------------------------------
    # RESEARCH OBJECTIVE
    # --------------------------------------------------------

    st.subheader("Research Objective")

    st.write(
        """
        The demonstrator evaluates privacy-preserving machine learning
        using Fully Homomorphic Encryption. The experiments compare
        conventional plaintext inference with FHE inference while
        measuring predictive performance, latency, memory consumption,
        simulation-to-real agreement, and computational overhead.
        """
    )

    # --------------------------------------------------------
    # ARCHITECTURE
    # --------------------------------------------------------

    st.subheader("FHE Client–Server Architecture")

    st.code(
        """
User Data
    │
    ▼
┌─────────────────────┐
│ Client              │
│ Quantization        │
│ Encryption          │
└─────────┬───────────┘
          │
          │ Ciphertext
          ▼
┌─────────────────────┐
│ Server              │
│ FHE Inference       │
│ on Encrypted Data   │
└─────────┬───────────┘
          │
          │ Encrypted Result
          ▼
┌─────────────────────┐
│ Client              │
│ Decryption          │
│ Dequantization      │
└─────────┬───────────┘
          │
          ▼
      Prediction
        """
    )

    st.divider()

    st.subheader("Datasets")

    st.write(
        ", ".join(sorted(master["dataset"].unique()))
    )

    st.subheader("Models")

    st.write(
        ", ".join(sorted(master["model"].unique()))
    )


# ============================================================
# EXPERIMENTAL DASHBOARD
# ============================================================

elif page == "📊 Experimental Dashboard":

    st.title("📊 Experimental Dashboard")

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    ds = st.multiselect(
        "Datasets",
        options=sorted(master["dataset"].unique()),
        default=sorted(master["dataset"].unique())
    )

    models = st.multiselect(
        "Models",
        options=sorted(master["model"].unique()),
        default=sorted(master["model"].unique())
    )

    df = master[
        master["dataset"].isin(ds)
        & master["model"].isin(models)
    ].copy()

    if df.empty:

        st.warning(
            "No configuration matches the selected filters."
        )

        st.stop()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Configurations",
        len(df)
    )

    c2.metric(
        "Mean Plaintext Accuracy",
        pct(df["acc_plain"].mean())
    )

    c3.metric(
        "Mean FHE Simulated Accuracy",
        pct(df["acc_fhe_simulate_full"].mean())
    )

    c4.metric(
        "Mean Simulation / Real Agreement",
        pct(df["agreement_simulate_vs_real"].mean())
    )

    st.divider()

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    st.subheader("Main FHE Experimental Results")

    show = df[
        [
            "dataset",
            "model",
            "acc_plain",
            "acc_fhe_simulate_full",
            "delta_acc_fhe_minus_plain",
            "acc_fhe_real_subsample",
            "agreement_simulate_vs_real",
            "latency_fhe_mean_s",
            "latency_ci95_low_s",
            "latency_ci95_high_s",
            "memory_peak_rss_mb",
            "overhead_ratio"
        ]
    ].copy()

    st.dataframe(
        show.style.format(
            {
                "acc_plain": "{:.4f}",
                "acc_fhe_simulate_full": "{:.4f}",
                "delta_acc_fhe_minus_plain": "{:+.4f}",
                "acc_fhe_real_subsample": "{:.4f}",
                "agreement_simulate_vs_real": "{:.4f}",
                "latency_fhe_mean_s": "{:.4f}",
                "latency_ci95_low_s": "{:.4f}",
                "latency_ci95_high_s": "{:.4f}",
                "memory_peak_rss_mb": "{:.1f}",
                "overhead_ratio": "{:.0f}x"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # LATENCY CHART
    # --------------------------------------------------------

    st.subheader("FHE Latency by Dataset and Model")

    latency_chart = df.pivot(
        index="dataset",
        columns="model",
        values="latency_fhe_mean_s"
    )

    st.bar_chart(latency_chart)

    st.caption(
        "Latency corresponds to the mean FHE execution time "
        "recorded in the thesis notebook."
    )

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    st.subheader("Peak Memory Consumption")

    memory_chart = df.pivot(
        index="dataset",
        columns="model",
        values="memory_peak_rss_mb"
    )

    st.bar_chart(memory_chart)


# ============================================================
# PLAINTEXT VS FHE
# ============================================================

elif page == "📈 Plaintext vs FHE":

    st.title("📈 Plaintext vs FHE Comparison")

    metric = st.selectbox(
        "Metric",
        [
            "Accuracy",
            "F1-score"
        ]
    )

    if metric == "Accuracy":

        tmp = master[
            [
                "dataset",
                "model",
                "acc_plain",
                "acc_fhe_simulate_full"
            ]
        ].melt(
            ["dataset", "model"],
            var_name="mode",
            value_name="value"
        )

    else:

        tmp = master[
            [
                "dataset",
                "model",
                "f1_plain",
                "f1_fhe_simulate_full"
            ]
        ].melt(
            ["dataset", "model"],
            var_name="mode",
            value_name="value"
        )

    tmp["configuration"] = (
        tmp["dataset"]
        + " — "
        + tmp["model"]
    )

    st.bar_chart(
        tmp.pivot(
            index="configuration",
            columns="mode",
            values="value"
        )
    )

    st.divider()

    # --------------------------------------------------------
    # REAL FHE VS SIMULATION
    # --------------------------------------------------------

    st.subheader(
        "FHE Simulation vs Real Execution"
    )

    comparison = master[
        [
            "dataset",
            "model",
            "acc_fhe_simulate_full",
            "acc_fhe_real_subsample",
            "f1_fhe_simulate_full",
            "f1_fhe_real_subsample",
            "agreement_simulate_vs_real"
        ]
    ].copy()

    st.dataframe(
        comparison.style.format(
            {
                "acc_fhe_simulate_full": "{:.4f}",
                "acc_fhe_real_subsample": "{:.4f}",
                "f1_fhe_simulate_full": "{:.4f}",
                "f1_fhe_real_subsample": "{:.4f}",
                "agreement_simulate_vs_real": "{:.4f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # LATENCY
    # --------------------------------------------------------

    st.subheader(
        "FHE Latency and 95% Confidence Interval"
    )

    latency_table = master[
        [
            "dataset",
            "model",
            "latency_fhe_mean_s",
            "latency_fhe_std_s",
            "latency_ci95_low_s",
            "latency_ci95_high_s"
        ]
    ].copy()

    st.dataframe(
        latency_table.style.format(
            {
                "latency_fhe_mean_s": "{:.4f}",
                "latency_fhe_std_s": "{:.4f}",
                "latency_ci95_low_s": "{:.4f}",
                "latency_ci95_high_s": "{:.4f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        """
        Simulated FHE accuracy and F1-score are evaluated on the
        complete test set. Real FHE metrics are evaluated on the
        stratified subsample recorded in the notebook.
        """
    )


# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

elif page == "🔬 Sensitivity Analysis":

    st.title("🔬 FHE Sensitivity Analysis")

    section = st.radio(
        "Analysis",
        [
            "Tree Depth",
            "Number of Trees",
            "Quantization Bits"
        ],
        horizontal=True
    )

    # --------------------------------------------------------
    # TREE DEPTH
    # --------------------------------------------------------

    if section == "Tree Depth":

        datasets_depth = st.multiselect(
            "Datasets",
            options=sorted(depth["dataset"].unique()),
            default=sorted(depth["dataset"].unique()),
            key="depth_datasets"
        )

        d = depth[
            depth["dataset"].isin(datasets_depth)
        ].copy()

        st.subheader("FHE Latency vs Tree Depth")

        latency = d.pivot(
            index="max_depth",
            columns="dataset",
            values="latency_s"
        )

        st.line_chart(latency)

        if "accuracy" in d.columns:

            st.subheader(
                "Simulated FHE Accuracy vs Tree Depth"
            )

            accuracy = d.pivot(
                index="max_depth",
                columns="dataset",
                values="accuracy"
            )

            st.line_chart(accuracy)

        st.subheader("Results")

        st.dataframe(
            d,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # NUMBER OF TREES
    # --------------------------------------------------------

    elif section == "Number of Trees":

        st.subheader(
            "FHE Latency vs Number of Trees"
        )

        latency = trees.pivot(
            index="n_estimators",
            columns="dataset",
            values="latency_s"
        )

        st.line_chart(latency)

        if "accuracy" in trees.columns:

            st.subheader(
                "Simulated FHE Accuracy vs Number of Trees"
            )

            accuracy = trees.pivot(
                index="n_estimators",
                columns="dataset",
                values="accuracy"
            )

            st.line_chart(accuracy)

        st.dataframe(
            trees,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # QUANTIZATION
    # --------------------------------------------------------

    else:

        st.subheader(
            "FHE Latency vs Quantization Bits"
        )

        latency = bits.pivot(
            index="n_bits",
            columns="dataset",
            values="latency_s"
        )

        st.line_chart(latency)

        if "accuracy" in bits.columns:

            st.subheader(
                "Simulated FHE Accuracy vs Quantization Bits"
            )

            accuracy = bits.pivot(
                index="n_bits",
                columns="dataset",
                values="accuracy"
            )

            st.line_chart(accuracy)

        st.dataframe(
            bits,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# CLIENT–SERVER
# ============================================================

elif page == "🖥️ Client–Server Deployment":

    st.title("🖥️ Client–Server FHE Deployment")

    st.write(
        """
        This experiment demonstrates a two-process FHE deployment
        using a client and a server. The client encrypts the input,
        the server performs FHE inference on encrypted data, and
        the client decrypts the response.
        """
    )

    st.info(
        """
        Configuration: WDBC + Decision Tree,
        max_depth=5, n_bits=6.
        This is a separate deployment experiment from the
        main 15-configuration FHE evaluation.
        """
    )

    # --------------------------------------------------------
    # FIXED EXPERIMENT VALUES
    # --------------------------------------------------------

    a, b, c, d = st.columns(4)

    a.metric(
        "Encryption",
        "12.6 ms"
    )

    b.metric(
        "Server FHE",
        "3.847 s"
    )

    c.metric(
        "Decryption",
        "3.5 ms"
    )

    d.metric(
        "End-to-End",
        "7.514 s"
    )

    st.divider()

    st.subheader("Client–Server Flow")

    st.code(
        """
CLIENT
  │
  ├── Quantize
  ├── Encrypt
  └── Serialize
        │
        │  Encrypted request
        ▼
SERVER
  │
  └── FHE inference
        │
        │  Encrypted response
        ▼
CLIENT
  │
  ├── Deserialize
  ├── Decrypt
  └── Dequantize
        │
        ▼
Prediction
        """
    )

    st.subheader("Recorded Deployment Results")

    st.dataframe(
        deploy,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        """
        The network and end-to-end measurements shown above come
        from the two-process client–server demonstrator. The
        'Total no network' value represents encryption time plus
        server FHE inference time plus decryption time.
        """
    )


# ============================================================
# NEURAL PREPROCESSING
# ============================================================

elif page == "🧠 Neural Preprocessing":

    st.title("🧠 Neural Preprocessing + FHE")

    st.write(
        """
        This section presents the neural preprocessing experiment
        recorded in the thesis notebook. The autoencoder uses a
        compressed latent representation before the downstream
        tree-based evaluation.
        """
    )

    st.subheader("Neural Preprocessing Results")

    st.dataframe(
        neural,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # RAW VS BOTTLENECK
    # --------------------------------------------------------

    if {
        "dataset",
        "latency_raw_s",
        "latency_bottleneck_s"
    }.issubset(neural.columns):

        st.subheader(
            "Raw Input vs Bottleneck Latency"
        )

        neural_chart = neural.set_index(
            "dataset"
        )[
            [
                "latency_raw_s",
                "latency_bottleneck_s"
            ]
        ]

        st.bar_chart(neural_chart)

    # --------------------------------------------------------
    # MLP FHE
    # --------------------------------------------------------

    st.subheader("FHE-MLP Results")

    st.dataframe(
        mlp,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        """
        The MLP-FHE results shown here are the results recorded
        in the corrected thesis notebook. They are presented
        separately from the main tree-based FHE experiments.
        """
    )


# ============================================================
# FHE + DIFFERENTIAL PRIVACY
# ============================================================

elif page == "🔒 FHE + Differential Privacy":

    st.title(
        "🔒 FHE + Differential Privacy"
    )

    st.write(
        """
        This section presents the illustrative Monte Carlo
        Differential Privacy simulation recorded in the notebook.
        """
    )

    st.info(
        """
        This experiment is presented as an illustrative simulation,
        not as a formal proof of an (ε, δ)-Differential Privacy
        guarantee for the FHE model.
        """
    )

    st.subheader("Simulation Results")

    st.dataframe(
        dp,
        use_container_width=True,
        hide_index=True
    )

    if "dataset" in dp.columns:

        sel = st.selectbox(
            "Dataset",
            sorted(dp["dataset"].unique())
        )

        d = dp[
            dp["dataset"] == sel
        ].copy()

        if {
            "epsilon",
            "model",
            "acc_dp_mean"
        }.issubset(d.columns):

            st.subheader(
                "Accuracy under DP Noise"
            )

            chart = d.pivot(
                index="epsilon",
                columns="model",
                values="acc_dp_mean"
            )

            st.line_chart(chart)

    st.warning(
        """
        The notebook treats this as an illustrative Monte Carlo
        simulation. It should not be interpreted as a formal
        (ε, δ)-DP proof. The sensitivity assumption and output
        perturbation mechanism are simplified experimental
        assumptions.
        """
    )


# ============================================================
# METHODOLOGY
# ============================================================

else:

    st.title("📚 Methodology")

    st.markdown(
        """
### Datasets

WDBC, Spambase, Adult, Pima Diabetes, and Heart Disease.

### Data Split

80/20 train-test split, `random_state=42`, with stratification.

### Plaintext Baseline

- Decision Tree: `max_depth=5`
- Random Forest: `n_estimators=50`, `max_depth=5`
- XGBoost: `n_estimators=50`, `max_depth=5`

### Main FHE Experiments

- Concrete-ML 1.9.0
- Decision Tree: `max_depth=4`
- Random Forest: `n_estimators=15`, `max_depth=4`
- XGBoost: `n_estimators=15`, `max_depth=4`
- Quantization: `n_bits=5`
- Calibration: 50 training observations
- Real FHE evaluation: 30 stratified observations
- FHE latency: 3 repetitions
- Simulated FHE accuracy/F1: complete test set
- Real FHE accuracy/F1: recorded subsample
- Simulation/real prediction agreement
- 95% confidence interval for FHE latency
- Peak RSS memory

### Sensitivity Analysis

- Tree depth: 3, 5, 7, 10
- Number of trees: 10, 50, 100
- Quantization bits: 2, 4, 6, 8

### Client–Server Deployment

WDBC + Decision Tree, `max_depth=5`, `n_bits=6`.

### Neural Preprocessing

Autoencoder-based preprocessing with a compressed latent representation,
followed by evaluation of the FHE-compatible MLP configuration recorded
in the thesis notebook.

### FHE + Differential Privacy

Illustrative Monte Carlo output-perturbation simulation.

This section is presented as an experimental simulation and not as a
formal proof of an `(ε, δ)`-Differential Privacy guarantee.

### Data Provenance

All displayed experimental results are loaded from the CSV files
generated from the thesis notebook.

The application does not re-run the computationally expensive FHE
experiments automatically.
        """
    )

    st.subheader("Available Result Files")

    files = sorted(
        [
            p.name
            for p in DATA.iterdir()
            if p.is_file()
        ]
    )

    st.write(files)

    st.divider()

    st.caption(
        "FHE Thesis Demonstrator • Experimental results from the thesis notebook"
    )
