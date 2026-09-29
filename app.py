import streamlit as st
import pandas as pd
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
# DATA LOADING
# ============================================================

@st.cache_data
def load(name):
    path = DATA / name

    if not path.exists():
        st.error(f"Required file not found: {path}")
        st.stop()

    return pd.read_csv(path)


# Main experimental results
master = load("master_results_latest.csv")

# Raw FHE results
fhe = load("df_fhe.csv")

# Sensitivity experiments
depth = load("depth_results.csv")
trees = load("ntrees_results.csv")
bits = load("bits_results.csv")

# Other experiments
dp = load("dp_simulation_results.csv")
neural = load("neural_preprocessing_results.csv")
mlp = load("mlp_fhe_results.csv")
deploy = load("client_server_results.csv")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.2f}%"


def safe_mean(series):
    if series.empty:
        return None

    numeric = pd.to_numeric(series, errors="coerce").dropna()

    if numeric.empty:
        return None

    return numeric.mean()


def format_dataframe(df, formats):
    """
    Apply formatting only to columns that actually exist.
    This prevents Streamlit/Pandas formatting errors.
    """

    valid_formats = {
        col: fmt
        for col, fmt in formats.items()
        if col in df.columns
    }

    if not valid_formats:
        return df

    return df.style.format(valid_formats)


# ============================================================
# VERIFY MAIN DATA
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
    "overhead_ratio",
]


missing_master = [
    col for col in required_master_columns
    if col not in master.columns
]


if missing_master:

    st.error(
        "The file `master_results_latest.csv` is missing the following "
        "required columns:"
    )

    for col in missing_master:
        st.write(f"- `{col}`")

    st.info(
        "Please make sure that `data/master_results_latest.csv` is the "
        "CSV generated from the corrected thesis notebook."
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
        "📊 Dashboard",
        "📈 Plaintext vs FHE",
        "🔬 Sensitivity Analysis",
        "🖥️ Client–Server",
        "🧠 Neural Preprocessing",
        "🔒 FHE + DP",
        "📚 Methodology",
    ]
)

st.sidebar.caption(
    "Experimental results extracted from the thesis notebook"
)

st.sidebar.caption(
    "Concrete-ML 1.9.0 • Main FHE experiment: n_bits=5"
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title("🔐 FHE Privacy-Preserving Machine Learning")

    st.subheader(
        "Interactive MSc Thesis Experimental Demonstrator"
    )

    st.write(
        """
        This application presents the experimental results obtained
        from the thesis notebook on privacy-preserving machine learning
        using Fully Homomorphic Encryption (FHE).

        The interface distinguishes between FHE simulation, real FHE
        execution, sensitivity experiments, client–server deployment,
        neural preprocessing, and the illustrative Differential
        Privacy simulation.
        """
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Datasets",
        master["dataset"].nunique()
    )

    b.metric(
        "Models",
        master["model"].nunique()
    )

    c.metric(
        "FHE Configurations",
        len(master)
    )

    mean_latency = safe_mean(
        master["latency_fhe_mean_s"]
    )

    d.metric(
        "Mean FHE Latency",
        f"{mean_latency:.3f} s"
        if mean_latency is not None
        else "N/A"
    )

    st.divider()

    st.subheader("Thesis Scope")

    st.markdown(
        """
        **Datasets**

        - WDBC
        - Spambase
        - Adult
        - Pima Diabetes
        - Heart Disease

        **Models**

        - Decision Tree
        - Random Forest
        - XGBoost

        **Main FHE configuration**

        - Concrete-ML 1.9.0
        - `n_bits = 5`
        - real FHE evaluation on a subsample
        - repeated latency measurements
        """
    )

    st.subheader("Architecture")

    st.code(
        """
User Data
    ↓
FHE Client
    ↓
Quantization + Encryption
    ↓
Network
    ↓
FHE Server
    ↓
Inference on Ciphertext
    ↓
Network
    ↓
FHE Client
    ↓
Decryption + Dequantization
    ↓
Prediction
        """
    )

    st.info(
        "The displayed experimental values are loaded from the CSV "
        "results generated by the thesis notebook. The Streamlit "
        "application does not automatically re-run the computationally "
        "expensive FHE experiments."
    )


# ============================================================
# DASHBOARD
# ============================================================

elif page == "📊 Dashboard":

    st.title("📊 Experimental Dashboard")

    st.write(
        "Interactive overview of the 15 main FHE configurations."
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    datasets_available = list(
        master["dataset"].dropna().unique()
    )

    models_available = list(
        master["model"].dropna().unique()
    )

    ds = st.multiselect(
        "Datasets",
        datasets_available,
        default=datasets_available
    )

    models = st.multiselect(
        "Models",
        models_available,
        default=models_available
    )

    df = master[
        master["dataset"].isin(ds)
        &
        master["model"].isin(models)
    ].copy()

    if df.empty:

        st.warning(
            "No configuration matches the selected filters."
        )

        st.stop()

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Configurations",
        len(df)
    )

    mean_fhe_accuracy = safe_mean(
        df["acc_fhe_simulate_full"]
    )

    c2.metric(
        "Average FHE Accuracy",
        pct(mean_fhe_accuracy)
        if mean_fhe_accuracy is not None
        else "N/A"
    )

    mean_plain_accuracy = safe_mean(
        df["acc_plain"]
    )

    c3.metric(
        "Average Plaintext Accuracy",
        pct(mean_plain_accuracy)
        if mean_plain_accuracy is not None
        else "N/A"
    )

    mean_agreement = safe_mean(
        df["agreement_simulate_vs_real"]
    )

    c4.metric(
        "Average Simulation / Real Agreement",
        pct(mean_agreement)
        if mean_agreement is not None
        else "N/A"
    )

    st.divider()

    # --------------------------------------------------------
    # MAIN RESULTS TABLE
    # --------------------------------------------------------

    st.subheader("Main FHE Results")

    display_columns = [
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
        "overhead_ratio",
    ]

    show = df[display_columns].copy()

    styled_show = format_dataframe(
        show,
        {
            "acc_plain": "{:.4f}",
            "acc_fhe_simulate_full": "{:.4f}",
            "delta_acc_fhe_minus_plain": "{:+.4f}",
            "acc_fhe_real_subsample": "{:.4f}",
            "agreement_simulate_vs_real": "{:.2%}",
            "latency_fhe_mean_s": "{:.4f}",
            "latency_ci95_low_s": "{:.4f}",
            "latency_ci95_high_s": "{:.4f}",
            "memory_peak_rss_mb": "{:.1f}",
            "overhead_ratio": "{:.0f}x",
        }
    )

    st.dataframe(
        styled_show,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # LATENCY
    # --------------------------------------------------------

    st.subheader(
        "FHE Latency by Dataset and Model"
    )

    chart = df.pivot(
        index="dataset",
        columns="model",
        values="latency_fhe_mean_s"
    )

    st.bar_chart(chart)

    st.caption(
        "Latency corresponds to the recorded mean FHE latency "
        "for each dataset/model configuration."
    )


# ============================================================
# PLAINTEXT VS FHE
# ============================================================

elif page == "📈 Plaintext vs FHE":

    st.title(
        "📈 Plaintext vs FHE Comparison"
    )

    metric = st.selectbox(
        "Metric",
        [
            "Accuracy",
            "F1-score"
        ]
    )

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    if metric == "Accuracy":

        tmp = master[
            [
                "dataset",
                "model",
                "acc_plain",
                "acc_fhe_simulate_full",
            ]
        ].melt(
            [
                "dataset",
                "model"
            ],
            var_name="mode",
            value_name="value"
        )

        tmp["mode"] = tmp["mode"].replace(
            {
                "acc_plain": "Plaintext",
                "acc_fhe_simulate_full": "FHE Simulation",
            }
        )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    else:

        tmp = master[
            [
                "dataset",
                "model",
                "f1_plain",
                "f1_fhe_simulate_full",
            ]
        ].melt(
            [
                "dataset",
                "model"
            ],
            var_name="mode",
            value_name="value"
        )

        tmp["mode"] = tmp["mode"].replace(
            {
                "f1_plain": "Plaintext",
                "f1_fhe_simulate_full": "FHE Simulation",
            }
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

    # --------------------------------------------------------
    # LATENCY + CI
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
            "latency_ci95_high_s",
        ]
    ].copy()

    styled_latency = format_dataframe(
        latency_table,
        {
            "latency_fhe_mean_s": "{:.4f}",
            "latency_fhe_std_s": "{:.4f}",
            "latency_ci95_low_s": "{:.4f}",
            "latency_ci95_high_s": "{:.4f}",
        }
    )

    st.dataframe(
        styled_latency,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "FHE simulation accuracy/F1-score is evaluated on the complete "
        "test set. Real FHE accuracy/F1-score is evaluated on the "
        "recorded real-FHE subsample."
    )

    # --------------------------------------------------------
    # REAL FHE RESULTS
    # --------------------------------------------------------

    st.subheader(
        "Simulation vs Real FHE"
    )

    real_table = master[
        [
            "dataset",
            "model",
            "acc_fhe_simulate_full",
            "acc_fhe_real_subsample",
            "f1_fhe_simulate_full",
            "f1_fhe_real_subsample",
            "agreement_simulate_vs_real",
        ]
    ].copy()

    styled_real = format_dataframe(
        real_table,
        {
            "acc_fhe_simulate_full": "{:.4f}",
            "acc_fhe_real_subsample": "{:.4f}",
            "f1_fhe_simulate_full": "{:.4f}",
            "f1_fhe_real_subsample": "{:.4f}",
            "agreement_simulate_vs_real": "{:.2%}",
        }
    )

    st.dataframe(
        styled_real,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

elif page == "🔬 Sensitivity Analysis":

    st.title("🔬 Sensitivity Analysis")

    section = st.radio(
        "Analysis",
        [
            "Tree Depth",
            "Number of Trees",
            "Quantization"
        ],
        horizontal=True
    )

    # --------------------------------------------------------
    # TREE DEPTH
    # --------------------------------------------------------

    if section == "Tree Depth":

        selected_datasets = st.multiselect(
            "Datasets",
            list(depth["dataset"].dropna().unique()),
            default=list(
                depth["dataset"].dropna().unique()
            ),
            key="depth_datasets"
        )

        d = depth[
            depth["dataset"].isin(selected_datasets)
        ].copy()

        if d.empty:

            st.warning(
                "No results available for the selected datasets."
            )

        else:

            st.subheader(
                "FHE Latency vs Tree Depth"
            )

            st.line_chart(
                d.pivot(
                    index="max_depth",
                    columns="dataset",
                    values="latency_s"
                )
            )

            if "accuracy" in d.columns:

                st.subheader(
                    "Simulated FHE Accuracy"
                )

                st.line_chart(
                    d.pivot(
                        index="max_depth",
                        columns="dataset",
                        values="accuracy"
                    )
                )

            st.subheader(
                "Tree Depth Results"
            )

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

        st.line_chart(
            trees.pivot(
                index="n_estimators",
                columns="dataset",
                values="latency_s"
            )
        )

        if "accuracy" in trees.columns:

            st.subheader(
                "Simulated FHE Accuracy"
            )

            st.line_chart(
                trees.pivot(
                    index="n_estimators",
                    columns="dataset",
                    values="accuracy"
                )
            )

        st.subheader(
            "Number of Trees Results"
        )

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

        st.line_chart(
            bits.pivot(
                index="n_bits",
                columns="dataset",
                values="latency_s"
            )
        )

        if "accuracy" in bits.columns:

            st.subheader(
                "Simulated FHE Accuracy"
            )

            st.line_chart(
                bits.pivot(
                    index="n_bits",
                    columns="dataset",
                    values="accuracy"
                )
            )

        st.subheader(
            "Quantization Results"
        )

        st.dataframe(
            bits,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# CLIENT–SERVER
# ============================================================

elif page == "🖥️ Client–Server":

    st.title(
        "🖥️ FHE Client–Server Deployment"
    )

    st.write(
        """
        Separate client–server FHE deployment demonstration based on
        the WDBC dataset and a Decision Tree configuration.
        """
    )

    st.write(
        """
        **Configuration:** WDBC + Decision Tree,
        `max_depth=5`, `n_bits=6`.
        """
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Encryption",
        "12.6 ms"
    )

    b.metric(
        "FHE Server",
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

    st.subheader(
        "Execution Flow"
    )

    st.code(
        """
CLIENT
  quantize_encrypt_serialize()
       │
       │ request = 984 bytes
       ▼
SERVER
  server.run(ciphertext, evaluation_keys)
       │
       │ response = 33056 bytes
       ▼
CLIENT
  deserialize_decrypt_dequantize()
        """
    )

    st.subheader(
        "Client–Server Recorded Results"
    )

    st.dataframe(
        deploy,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "The network and end-to-end times shown above come from the "
        "two-process client–server demonstrator. The 'Total no network' "
        "calculation adds encryption, server inference, and decryption."
    )


# ============================================================
# NEURAL PREPROCESSING
# ============================================================

elif page == "🧠 Neural Preprocessing":

    st.title(
        "🧠 Neural Preprocessing + Tree"
    )

    st.write(
        """
        Neural preprocessing experiment using an autoencoder-based
        compressed representation before downstream evaluation.
        """
    )

    st.subheader(
        "Neural Preprocessing Results"
    )

    st.dataframe(
        neural,
        use_container_width=True,
        hide_index=True
    )

    if {
        "dataset",
        "latency_raw_s",
        "latency_bottleneck_s"
    }.issubset(neural.columns):

        st.subheader(
            "Raw vs Bottleneck Latency"
        )

        st.bar_chart(
            neural.set_index("dataset")[
                [
                    "latency_raw_s",
                    "latency_bottleneck_s"
                ]
            ]
        )

    st.subheader(
        "FHE-MLP Baseline"
    )

    st.dataframe(
        mlp,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The MLP results are displayed separately from the main "
        "tree-based FHE experiments."
    )


# ============================================================
# FHE + DIFFERENTIAL PRIVACY
# ============================================================

elif page == "🔒 FHE + DP":

    st.title(
        "🔒 FHE + Differential Privacy"
    )

    st.write(
        """
        Illustrative Monte Carlo simulation of an output perturbation
        mechanism.
        """
    )

    st.dataframe(
        dp,
        use_container_width=True,
        hide_index=True
    )

    if "dataset" in dp.columns:

        selected_dataset = st.selectbox(
            "Dataset",
            list(dp["dataset"].dropna().unique())
        )

        d = dp[
            dp["dataset"] == selected_dataset
        ].copy()

        if {
            "epsilon",
            "model",
            "acc_dp_mean"
        }.issubset(d.columns):

            st.subheader(
                "Differential Privacy Simulation"
            )

            st.line_chart(
                d.pivot(
                    index="epsilon",
                    columns="model",
                    values="acc_dp_mean"
                )
            )

    st.warning(
        """
        This section is presented as an illustrative simulation.
        It is not a formal proof of an (ε,δ)-Differential Privacy
        guarantee for the FHE model.
        """
    )


# ============================================================
# METHODOLOGY
# ============================================================

else:

    st.title(
        "📚 Methodology"
    )

    st.markdown(
        """
### Datasets

The main experimental study covers:

- WDBC
- Spambase
- Adult
- Pima Diabetes
- Heart Disease

### Models

The main tree-based models are:

- Decision Tree
- Random Forest
- XGBoost

### Main FHE Experiments

The main FHE experimental results are loaded from
`df_fhe.csv` and `master_results_latest.csv`.

The reported variables include:

- plaintext accuracy;
- plaintext F1-score;
- simulated FHE accuracy;
- simulated FHE F1-score;
- real FHE accuracy;
- real FHE F1-score;
- simulation/real agreement;
- mean FHE latency;
- FHE latency standard deviation;
- 95% latency confidence interval;
- peak RSS memory;
- accuracy difference;
- computational overhead.

### Main FHE Configuration

The main experiment uses:

- Concrete-ML 1.9.0;
- `n_bits = 5`;
- Decision Tree: `max_depth = 4`;
- Random Forest: `n_estimators = 15`, `max_depth = 4`;
- XGBoost: `n_estimators = 15`, `max_depth = 4`;
- 50 calibration samples;
- 30 real FHE evaluation samples;
- 3 latency repetitions.

### Sensitivity Analysis

The application presents sensitivity experiments for:

- tree depth;
- number of trees;
- quantization bits.

### Client–Server Demonstration

The separate client–server demonstration uses:

- WDBC;
- Decision Tree;
- `max_depth = 5`;
- `n_bits = 6`.

### Data Provenance

The Streamlit interface reads the experimental results from
the `data/` directory.

The original thesis notebook is the source of the experimental
results.

The application is intended for interactive presentation and
visualization rather than re-running the complete FHE benchmark.
"""
    )

    st.subheader(
        "Loaded Result Files"
    )

    files = [
        "master_results_latest.csv",
        "df_fhe.csv",
        "depth_results.csv",
        "ntrees_results.csv",
        "bits_results.csv",
        "dp_simulation_results.csv",
        "neural_preprocessing_results.csv",
        "mlp_fhe_results.csv",
        "client_server_results.csv",
    ]

    for filename in files:

        path = DATA / filename

        if path.exists():
            st.write(f"✅ `{filename}`")
        else:
            st.write(f"❌ `{filename}`")
