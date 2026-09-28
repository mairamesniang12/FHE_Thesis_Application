import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="FHE Thesis Demonstrator",
    page_icon="🔐",
    layout="wide"
)

ROOT = Path(__file__).parent
DATA = ROOT / "data"


@st.cache_data
def load(name):
    return pd.read_csv(DATA / name)


master = load("master_results.csv")
depth = load("depth_results.csv")
trees = load("ntrees_results.csv")
bits = load("bits_results.csv")
dp = load("dp_simulation_results.csv")
neural = load("neural_preprocessing_results.csv")
mlp = load("mlp_fhe_results.csv")
deploy = load("client_server_results.csv")


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
        "📚 Methodology"
    ]
)

st.sidebar.caption("Results extracted from the provided notebook")
st.sidebar.caption(
    "Concrete-ML 1.9.0 • n_bits=5 for the main experiment"
)


def pct(x):
    return f"{x * 100:.2f}%"


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title("🔐 FHE Privacy-Preserving Machine Learning")
    st.subheader("Interactive Thesis Demonstrator")

    st.write(
        "This application presents the experimental results recorded "
        "in the reference notebook, with an explicit distinction between "
        "FHE simulation, real FHE execution, and the Monte Carlo DP simulation."
    )

    a, b, c, d = st.columns(4)

    a.metric("Datasets", "5")
    b.metric("Models", "3")
    c.metric("FHE Experiments", "15")
    d.metric("FHE Repetitions", "3")

    st.divider()

    st.subheader("Architecture")

    st.code(
        """User Data
       ↓
FHE Client — Quantization + Encryption
       ↓
Network
       ↓
FHE Server — Inference on Ciphertext
       ↓
Network
       ↓
Client — Decryption
       ↓
Prediction"""
    )

    st.info(
        "The displayed metrics come from the provided notebook. "
        "The application does not automatically re-run the computationally "
        "expensive experiments."
    )


# ============================================================
# DASHBOARD
# ============================================================

elif page == "📊 Dashboard":

    st.title("📊 Experimental Dashboard")

    ds = st.multiselect(
        "Datasets",
        master.dataset.unique(),
        default=list(master.dataset.unique())
    )

    models = st.multiselect(
        "Models",
        master.model.unique(),
        default=list(master.model.unique())
    )

    df = master[
        master.dataset.isin(ds) &
        master.model.isin(models)
    ].copy()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Configurations",
        len(df)
    )

    c2.metric(
        "Average FHE Accuracy",
        pct(df.acc_fhe_sim.mean())
    )

    c3.metric(
        "Average Plaintext Accuracy",
        pct(df.acc_plain.mean())
    )

    show = df[
        [
            "dataset",
            "model",
            "acc_plain",
            "acc_fhe_sim",
            "delta_acc",
            "acc_fhe_real",
            "agreement",
            "latency_s",
            "ci_low_s",
            "ci_high_s",
            "memory_mb",
            "overhead_ratio"
        ]
    ].copy()

    st.dataframe(
        show.style.format(
            {
                "acc_plain": "{:.4f}",
                "acc_fhe_sim": "{:.4f}",
                "delta_acc": "{:+.4f}",
                "acc_fhe_real": "{:.4f}",
                "agreement": "{:.2f}",
                "latency_s": "{:.4f}",
                "ci_low_s": "{:.4f}",
                "ci_high_s": "{:.4f}",
                "memory_mb": "{:.1f}",
                "overhead_ratio": "{:.0f}x"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("FHE Latency by Dataset and Model")

    chart = df.pivot(
        index="dataset",
        columns="model",
        values="latency_s"
    )

    st.bar_chart(chart)


# ============================================================
# PLAINTEXT VS FHE
# ============================================================

elif page == "📈 Plaintext vs FHE":

    st.title("📈 Plaintext vs FHE Comparison")

    metric = st.selectbox(
        "Metric",
        ["Accuracy", "F1-score"]
    )

    if metric == "Accuracy":

        tmp = master[
            [
                "dataset",
                "model",
                "acc_plain",
                "acc_fhe_sim"
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
                "f1_fhe_sim"
            ]
        ].melt(
            ["dataset", "model"],
            var_name="mode",
            value_name="value"
        )

    tmp["configuration"] = (
        tmp.dataset + " — " + tmp.model
    )

    st.bar_chart(
        tmp.pivot(
            index="configuration",
            columns="mode",
            values="value"
        )
    )

    st.subheader(
        "FHE Latency and Confidence Interval"
    )

    st.dataframe(
        master[
            [
                "dataset",
                "model",
                "latency_s",
                "latency_std_s",
                "ci_low_s",
                "ci_high_s"
            ]
        ].style.format("{:.4f}"),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Simulated FHE accuracy is calculated on the complete test set; "
        "real FHE accuracy is reported on a stratified subsample of 30 observations."
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

    if section == "Tree Depth":

        selected_datasets = st.multiselect(
            "Datasets",
            depth.dataset.unique(),
            default=list(depth.dataset.unique()),
            key="depds"
        )

        d = depth[
            depth.dataset.isin(selected_datasets)
        ]

        st.line_chart(
            d.pivot(
                index="max_depth",
                columns="dataset",
                values="latency_s"
            )
        )

        st.subheader("Simulated FHE Accuracy")

        st.line_chart(
            d.pivot(
                index="max_depth",
                columns="dataset",
                values="accuracy"
            )
        )

        st.dataframe(
            d,
            use_container_width=True,
            hide_index=True
        )

    elif section == "Number of Trees":

        st.line_chart(
            trees.pivot(
                index="n_estimators",
                columns="dataset",
                values="latency_s"
            )
        )

        st.dataframe(
            trees,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.line_chart(
            bits.pivot(
                index="n_bits",
                columns="dataset",
                values="latency_s"
            )
        )

        st.subheader("Simulated FHE Accuracy")

        st.line_chart(
            bits.pivot(
                index="n_bits",
                columns="dataset",
                values="accuracy"
            )
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

    st.title("🖥️ FHE Client–Server Deployment")

    st.write(
        "WDBC + Decision Tree experiment, max_depth=5, n_bits=6, "
        "based on FHEModelDev/FHEModelClient/FHEModelServer."
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

    st.subheader("Execution Flow")

    st.code(
        """CLIENT
  quantize_encrypt_serialize()
       │  request = 984 bytes
       ▼
SERVER
  server.run(ciphertext, evaluation_keys)
       │  response = 33056 bytes
       ▼
CLIENT
  deserialize_decrypt_dequantize()
"""
    )

    st.dataframe(
        deploy,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "The network and end-to-end times above come from the "
        "two-process demonstrator. The 'Total no network' calculation "
        "adds encryption + server inference + decryption."
    )


# ============================================================
# NEURAL PREPROCESSING
# ============================================================

elif page == "🧠 Neural Preprocessing":

    st.title("🧠 Neural Preprocessing + Tree")

    st.write(
        "Autoencoder architecture: "
        "32-ReLU → 16-ReLU → linear bottleneck of dimension 8 "
        "→ symmetric decoder."
    )

    st.dataframe(
        neural,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Raw vs Bottleneck Latency")

    st.bar_chart(
        neural.set_index("dataset")[
            [
                "latency_raw_s",
                "latency_bottleneck_s"
            ]
        ]
    )

    st.subheader("FHE-MLP Baseline")

    st.dataframe(
        mlp,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "The notebook did not finalize the H1 comparison assembly in df_h1: "
        "the real MLP test was executed, but the aggregation failed with "
        "a NameError. Therefore, the MLP results are displayed separately "
        "without drawing a complete H1 comparison."
    )


# ============================================================
# FHE + DIFFERENTIAL PRIVACY
# ============================================================

elif page == "🔒 FHE + DP":

    st.title("🔒 FHE + Differential Privacy")

    st.write(
        "Illustrative Monte Carlo simulation of an output perturbation mechanism; "
        "20 repetitions per configuration, δ=10⁻⁵."
    )

    st.dataframe(
        dp,
        use_container_width=True,
        hide_index=True
    )

    sel = st.selectbox(
        "Dataset",
        dp.dataset.unique()
    )

    d = dp[
        dp.dataset == sel
    ].copy()

    st.line_chart(
        d.pivot(
            index="epsilon",
            columns="model",
            values="acc_dp_mean"
        )
    )

    st.warning(
        "The notebook explicitly requires this section to be presented "
        "as an illustrative simulation rather than as a formal proof of "
        "(ε,δ)-DP guarantees for the FHE model. Δf=1 is a simplifying "
        "assumption, and noise is added to the plaintext model probabilities."
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

### Plaintext Models

- Decision Tree: `max_depth=5`
- Random Forest: `n_estimators=50`, `max_depth=5`
- XGBoost: `n_estimators=50`, `max_depth=5`

### Main FHE Experiments

- Concrete-ML 1.9.0
- `n_bits=5`
- 3 latency repetitions
- 30 observations for real FHE execution
- Accuracy/F1 simulation on the complete test set
- Real accuracy/F1 on the subsample
- Simulation/real agreement
- 95% latency confidence interval
- Maximum RSS memory

### Sensitivity Analysis

- Tree depth: 3, 5, 7, 10
- Number of trees: 10, 50, 100
- Quantization bits: 2, 4, 6, 8

### Client–Server Deployment

WDBC + Decision Tree, `max_depth=5`, `n_bits=6`.

### Data Provenance

All data displayed by this interface are provided in the `data/`
directory, and the source notebook is included at the root of the project.
"""
    )

    st.subheader("Result Files")

    st.write(
        [
            p.name
            for p in DATA.iterdir()
        ]
    )
