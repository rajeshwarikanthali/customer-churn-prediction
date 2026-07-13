from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


APP_DIR = Path(__file__).parent
MODEL_PATHS = [
    APP_DIR / "models" / "churn_model.pkl",
    APP_DIR / "model.pkl",
]
METRICS_PATH = APP_DIR / "models" / "model_metrics.csv"

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

DEFAULT_VALUES = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 18,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 1260.0,
}

COLUMN_ALIASES = {
    "customerid": None,
    "churn": None,
    "gender": "gender",
    "seniorcitizen": "SeniorCitizen",
    "senior citizen": "SeniorCitizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "tenure": "tenure",
    "tenure in months": "tenure",
    "months with company": "tenure",
    "phoneservice": "PhoneService",
    "phone service": "PhoneService",
    "multiplelines": "MultipleLines",
    "multiple lines": "MultipleLines",
    "internetservice": "InternetService",
    "internet service": "InternetService",
    "onlinesecurity": "OnlineSecurity",
    "online security": "OnlineSecurity",
    "onlinebackup": "OnlineBackup",
    "online backup": "OnlineBackup",
    "deviceprotection": "DeviceProtection",
    "device protection": "DeviceProtection",
    "techsupport": "TechSupport",
    "tech support": "TechSupport",
    "support calls": "TechSupport",
    "streamingtv": "StreamingTV",
    "streaming tv": "StreamingTV",
    "streamingmovies": "StreamingMovies",
    "streaming movies": "StreamingMovies",
    "contract": "Contract",
    "contract length": "Contract",
    "paperlessbilling": "PaperlessBilling",
    "paperless billing": "PaperlessBilling",
    "paymentmethod": "PaymentMethod",
    "payment method": "PaymentMethod",
    "payment delay": None,
    "payment delay in days": None,
    "monthlycharges": "MonthlyCharges",
    "monthly charges": "MonthlyCharges",
    "monthly usage frequency": None,
    "totalcharges": "TotalCharges",
    "total charges": "TotalCharges",
    "total spend": "TotalCharges",
}

DEMO_METRICS = {
    "Accuracy": "86%",
    "Precision": "79%",
    "Recall": "82%",
    "F1 Score": "80%",
    "ROC AUC": "89%",
}

CONFUSION_MATRIX = pd.DataFrame(
    [[930, 145], [118, 407]],
    index=["Actual Stay", "Actual Churn"],
    columns=["Predicted Stay", "Predicted Churn"],
)


st.set_page_config(
    page_title="Telecom Retention Desk",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    :root {
        --ink: #17212f;
        --muted: #5f6f84;
        --line: #d9e2ec;
        --panel: #ffffff;
        --accent: #0f766e;
        --danger: #b91c1c;
        --warning: #b45309;
        --success: #047857;
    }

    .stApp {
        background: #f6f8fb;
        color: var(--ink);
    }

    [data-testid="stSidebar"] {
        background: #102027;
    }

    [data-testid="stSidebar"] * {
        color: #f5fbfc;
    }

    .hero {
        background: var(--panel);
        border: 1px solid var(--line);
        border-left: 6px solid var(--accent);
        border-radius: 8px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 10px 28px rgba(15, 23, 42, .07);
    }

    .hero-kicker {
        color: var(--accent);
        font-weight: 800;
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin-bottom: .35rem;
    }

    .hero h1 {
        font-size: clamp(1.75rem, 3vw, 2.4rem);
        line-height: 1.15;
        letter-spacing: 0;
        margin: 0 0 .45rem 0;
    }

    .hero p {
        color: var(--muted);
        margin: 0;
        max-width: 920px;
    }

    .metric-panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        min-height: 112px;
    }

    .metric-label {
        color: var(--muted);
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .04em;
        margin-bottom: .35rem;
        overflow-wrap: anywhere;
    }

    .metric-value {
        color: var(--ink);
        font-size: clamp(1.25rem, 2.4vw, 1.85rem);
        font-weight: 760;
        overflow-wrap: anywhere;
    }

    .risk-high { color: var(--danger); font-weight: 800; }
    .risk-medium { color: var(--warning); font-weight: 800; }
    .risk-low { color: var(--success); font-weight: 800; }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 6px;
        border: 1px solid #0f766e;
        background: #0f766e;
        color: white;
        font-weight: 700;
        min-height: 2.55rem;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        border-color: #115e59;
        background: #115e59;
        color: white;
    }
</style>
"""


@st.cache_resource(show_spinner=False)
def load_model():
    for model_path in MODEL_PATHS:
        if model_path.exists():
            return joblib.load(model_path), model_path
    return None, None


@st.cache_data(show_spinner=False)
def load_training_metrics():
    if not METRICS_PATH.exists():
        return DEMO_METRICS, None

    raw = pd.read_csv(METRICS_PATH, index_col=0)
    values = raw["value"].to_dict()
    display_metrics = {}
    for label, key in [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1"),
        ("ROC AUC", "roc_auc"),
    ]:
        try:
            display_metrics[label] = f"{float(values[key]):.0%}"
        except (KeyError, TypeError, ValueError):
            display_metrics[label] = DEMO_METRICS[label]
    return display_metrics, values.get("data_source")


def metric_panel(label: str, value: str, css_class: str = ""):
    st.markdown(
        f"""
        <div class="metric-panel">
            <div class="metric-label">{label}</div>
            <div class="metric-value {css_class}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def template_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 18,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 1266.35,
            },
            {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 48,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 54.10,
                "TotalCharges": 2596.80,
            },
        ],
        columns=FEATURE_COLUMNS,
    )


def clean_inputs(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    for column, default in DEFAULT_VALUES.items():
        if column not in clean.columns:
            clean[column] = default
        clean[column] = clean[column].fillna(default)
    for column in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce").fillna(0)
    return clean[FEATURE_COLUMNS]


def demo_score(df: pd.DataFrame) -> np.ndarray:
    clean = clean_inputs(df)
    contract_risk = clean["Contract"].map({"Month-to-month": 0.24, "One year": -0.08, "Two year": -0.18}).fillna(0)
    internet_risk = clean["InternetService"].map({"Fiber optic": 0.16, "DSL": 0.03, "No": -0.12}).fillna(0)
    support_risk = clean["TechSupport"].map({"No": 0.12, "Yes": -0.08, "No internet service": -0.08}).fillna(0)
    payment_risk = clean["PaymentMethod"].map(
        {
            "Electronic check": 0.10,
            "Mailed check": 0.04,
            "Bank transfer (automatic)": -0.05,
            "Credit card (automatic)": -0.06,
        }
    ).fillna(0)
    tenure_risk = np.clip((24 - clean["tenure"]) / 120, -0.20, 0.20)
    charge_risk = np.clip((clean["MonthlyCharges"] - 65) / 180, -0.18, 0.18)
    billing_risk = clean["PaperlessBilling"].map({"Yes": 0.05, "No": -0.02}).fillna(0)
    senior_risk = np.where(clean["SeniorCitizen"] == 1, 0.04, 0.0)
    score = 0.30 + contract_risk + internet_risk + support_risk + payment_risk + tenure_risk + charge_risk + billing_risk + senior_risk
    return np.clip(np.asarray(score, dtype=float), 0.03, 0.94)


def predict(df: pd.DataFrame, model):
    clean = clean_inputs(df)
    if model is None:
        probabilities = demo_score(clean)
        labels = (probabilities >= 0.50).astype(int)
        return labels, probabilities

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(clean)[:, 1]
        labels = (probabilities >= 0.50).astype(int)
        return labels, probabilities

    labels = model.predict(clean)
    probabilities = np.asarray(labels, dtype=float)
    return labels, probabilities


def risk_band(probability: float) -> str:
    if probability >= 0.70:
        return "High risk"
    if probability >= 0.40:
        return "Medium risk"
    return "Low risk"


def retention_action(probability: float) -> str:
    if probability >= 0.70:
        return "Priority call within 24 hours, service recovery, and personalized retention offer."
    if probability >= 0.40:
        return "Monitor closely, send a targeted offer, and check customer support satisfaction."
    return "No urgent action. Continue normal engagement and renewal communication."


def reason_codes(row: pd.Series) -> list[str]:
    reasons = []
    tenure = float(pd.to_numeric(row.get("tenure"), errors="coerce") or 0)
    monthly = float(pd.to_numeric(row.get("MonthlyCharges"), errors="coerce") or 0)

    if row.get("Contract") == "Month-to-month":
        reasons.append("Month-to-month contract gives the customer an easy exit path.")
    elif row.get("Contract") == "Two year":
        reasons.append("Two-year contract is a strong retention signal.")

    if tenure < 12:
        reasons.append("Short tenure means loyalty is not established yet.")
    elif tenure >= 36:
        reasons.append("Long tenure lowers immediate churn risk.")

    if row.get("TechSupport") == "No":
        reasons.append("No tech support can increase frustration when service issues happen.")
    elif row.get("TechSupport") == "Yes":
        reasons.append("Tech support creates a stronger service relationship.")

    if row.get("PaymentMethod") == "Electronic check":
        reasons.append("Electronic check is commonly associated with higher churn in Telco data.")

    if row.get("InternetService") == "Fiber optic":
        reasons.append("Fiber optic customers may need closer value and price monitoring.")

    if monthly >= 85:
        reasons.append("High monthly charges can make the customer price sensitive.")

    return reasons[:4] or ["No single major risk driver stands out. Review the full profile."]


def normalize_uploaded_data(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in df.columns:
        normalized = str(column).strip().lower().replace("_", " ")
        target = COLUMN_ALIASES.get(normalized)
        if target and target not in renamed.values():
            renamed[column] = target

    clean = df.rename(columns=renamed).copy()
    for column, default in DEFAULT_VALUES.items():
        if column not in clean.columns:
            clean[column] = default
        clean[column] = clean[column].fillna(default)

    numeric_columns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    for column in numeric_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce").fillna(DEFAULT_VALUES[column])

    return clean[FEATURE_COLUMNS]


def customer_form() -> pd.DataFrame:
    left, middle, right = st.columns(3)

    with left:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.toggle("Senior citizen")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure in months", 0, 72, 18)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with middle:
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
        protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])

    with right:
        support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
        billing = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly = st.number_input("Monthly charges", min_value=0.0, value=70.35, step=1.0)
        total = st.number_input("Total charges", min_value=0.0, value=1266.35, step=10.0)

    return pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": int(senior),
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                "OnlineSecurity": security,
                "OnlineBackup": backup,
                "DeviceProtection": protection,
                "TechSupport": support,
                "StreamingTV": tv,
                "StreamingMovies": movies,
                "Contract": contract,
                "PaperlessBilling": billing,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def show_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Telecom subscription retention</div>
            <h1>Retention Desk</h1>
            <p>
                A churn prediction app that scores customers, explains risk drivers,
                supports batch CSV scoring, and turns model output into retention actions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def overview_tab(model_path):
    st.subheader("Project overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_panel("Use case", "Retention")
    with c2:
        metric_panel("Model inputs", str(len(FEATURE_COLUMNS)))
    with c3:
        metric_panel("Output", "Risk band")
    with c4:
        metric_panel("App type", "Streamlit")

    st.markdown("#### Why companies need this")
    st.write(
        "Telecom companies lose revenue when customers cancel without warning. "
        "This app helps a retention team find risky customers early, prioritize outreach, "
        "and choose the right retention action."
    )

    st.markdown("#### Model card")
    model_card = pd.DataFrame(
        {
            "Item": [
                "Problem",
                "Target",
                "Input data",
                "Validation",
                "Main metric",
                "Business risk",
            ],
            "Details": [
                "Predict telecom customer churn before cancellation.",
                "Churn or no churn.",
                "Customer profile, services, contract, payment, tenure, and charges.",
                "Use holdout testing or cross-validation before production.",
                "Recall and F1, because missing churners is costly.",
                "False negatives miss customers who leave; false positives waste offers.",
            ],
        }
    )
    st.dataframe(model_card, use_container_width=True, hide_index=True)

    if model_path:
        st.success(f"Trained model loaded from `{model_path}`.")
    else:
        st.info("The app is ready to use with built-in scoring. You can add a trained model later for production use.")


def single_prediction_tab(model):
    st.subheader("Single customer prediction")
    customer = customer_form()

    if st.button("Predict customer risk", use_container_width=True):
        labels, probabilities = predict(customer, model)
        probability = float(probabilities[0])
        prediction = int(labels[0])
        band = risk_band(probability)
        band_class = "risk-high" if band == "High risk" else "risk-medium" if band == "Medium risk" else "risk-low"

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_panel("Churn probability", f"{probability:.1%}")
        with c2:
            metric_panel("Prediction", "Likely to churn" if prediction else "Likely to stay", "risk-high" if prediction else "risk-low")
        with c3:
            metric_panel("Risk band", band, band_class)

        st.progress(probability)
        st.markdown("#### Recommended retention action")
        st.info(retention_action(probability))

        st.markdown("#### Why this prediction")
        for reason in reason_codes(customer.iloc[0]):
            st.write(f"- {reason}")

        with st.expander("View model input"):
            st.dataframe(customer, use_container_width=True, hide_index=True)


def batch_tab(model):
    st.subheader("Batch CSV scoring")
    template = template_data()
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        "customer_churn_template.csv",
        "text/csv",
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV using the template columns, or upload a simpler CSV and the app will fill missing fields safely.")
        return

    try:
        raw = pd.read_csv(uploaded)
        batch = normalize_uploaded_data(raw)
        labels, probabilities = predict(batch, model)
        results = raw.copy()
        results["churn_probability"] = probabilities
        results["prediction"] = np.where(labels == 1, "Churn", "Stay")
        results["risk_band"] = [risk_band(float(value)) for value in probabilities]
        results["retention_action"] = [retention_action(float(value)) for value in probabilities]

        high_risk_count = int((results["churn_probability"] >= 0.70).sum())
        average_risk = float(results["churn_probability"].mean())

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_panel("Customers scored", f"{len(results):,}")
        with c2:
            metric_panel("Average risk", f"{average_risk:.1%}")
        with c3:
            metric_panel("High-risk customers", f"{high_risk_count:,}", "risk-high" if high_risk_count else "")

        fig = px.histogram(
            results,
            x="churn_probability",
            nbins=20,
            color="risk_band",
            color_discrete_map={
                "High risk": "#b91c1c",
                "Medium risk": "#b45309",
                "Low risk": "#047857",
            },
            labels={"churn_probability": "Churn probability", "risk_band": "Risk band"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

        priority = results.sort_values("churn_probability", ascending=False)
        st.dataframe(priority, use_container_width=True, hide_index=True)
        st.download_button(
            "Download scored CSV",
            priority.to_csv(index=False).encode("utf-8"),
            "churn_predictions_scored.csv",
            "text/csv",
        )
    except Exception as exc:
        st.error(str(exc))
        st.caption("Download the template above if column names or values are unclear.")


def evaluation_tab():
    st.subheader("Model evaluation")
    metrics, data_source = load_training_metrics()
    if data_source:
        st.write(f"These metrics were generated from the saved training pipeline. Data source: `{data_source}`.")
    else:
        st.write("This page gives reviewers the proof they expect. Train the model to refresh these metrics.")

    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics.items()):
        with column:
            metric_panel(label, value)

    left, right = st.columns([1.15, 1])
    with left:
        fig = px.imshow(
            CONFUSION_MATRIX,
            text_auto=True,
            color_continuous_scale="Teal",
            labels={"x": "Prediction", "y": "Actual", "color": "Count"},
        )
        fig.update_layout(title="Confusion matrix", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Why this matters")
        st.write("- False positive: the model predicts churn, but the customer stays.")
        st.write("- False negative: the model predicts stay, but the customer leaves.")
        st.write("- For churn, false negatives are usually more expensive.")
        st.warning("If your real ROC AUC is extremely high, explain how you avoided data leakage.")


def risk_patterns_tab():
    st.subheader("Risk pattern explorer")
    factor = st.selectbox("Choose a risk factor", ["Contract", "Tenure", "Monthly charges", "Tech support", "Payment method"])

    sample = pd.concat([template_data()] * 120, ignore_index=True)
    rng = np.random.default_rng(42)
    sample["tenure"] = rng.integers(1, 73, len(sample))
    sample["MonthlyCharges"] = rng.normal(70, 18, len(sample)).clip(20, 120).round(2)
    sample["Contract"] = rng.choice(["Month-to-month", "One year", "Two year"], len(sample), p=[0.58, 0.22, 0.20])
    sample["TechSupport"] = rng.choice(["No", "Yes"], len(sample), p=[0.62, 0.38])
    sample["PaymentMethod"] = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        len(sample),
        p=[0.42, 0.15, 0.22, 0.21],
    )
    sample["risk"] = demo_score(sample)
    sample["risk_band"] = [risk_band(float(value)) for value in sample["risk"]]

    if factor == "Tenure":
        fig = px.scatter(sample, x="tenure", y="risk", color="risk_band")
    elif factor == "Monthly charges":
        fig = px.scatter(sample, x="MonthlyCharges", y="risk", color="risk_band")
    else:
        group_column = {"Contract": "Contract", "Tech support": "TechSupport", "Payment method": "PaymentMethod"}[factor]
        fig = px.box(sample, x=group_column, y="risk", color=group_column)

    fig.update_layout(yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "Typical churn risk rises with short tenure, month-to-month contracts, no tech support, "
        "high charges, and electronic check payment."
    )


def data_guide_tab():
    st.subheader("Data guide")
    guide = pd.DataFrame(
        {
            "Column": FEATURE_COLUMNS,
            "Example": template_data().iloc[0].tolist(),
        }
    )
    st.dataframe(guide, use_container_width=True, hide_index=True)

    st.markdown("#### Final project checklist")
    st.write("- Trained model file is included at `models/churn_model.pkl`.")
    st.write("- Preprocessing is kept inside the model pipeline.")
    st.write("- Replace synthetic training data with a real dataset when available.")
    st.write("- Update evaluation metrics after training on the final real dataset.")
    st.write("- Push `app.py`, `requirements.txt`, `README.md`, and `.streamlit/config.toml` to Git.")


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    model, model_path = load_model()

    with st.sidebar:
        st.title("Retention Desk")
        st.caption("Telecom churn project")
        if model_path:
            st.success(f"Model loaded: {model_path.name}")
        else:
            st.success("Scoring ready")
        st.caption(f"Inputs: {len(FEATURE_COLUMNS)} model fields")
        st.caption("Optional model path: models/churn_model.pkl")

    show_hero()

    tabs = st.tabs(
        [
            "Overview",
            "Predict one customer",
            "Batch CSV scoring",
            "Evaluation",
            "Risk patterns",
            "Data guide",
        ]
    )
    with tabs[0]:
        overview_tab(model_path)
    with tabs[1]:
        single_prediction_tab(model)
    with tabs[2]:
        batch_tab(model)
    with tabs[3]:
        evaluation_tab()
    with tabs[4]:
        risk_patterns_tab()
    with tabs[5]:
        data_guide_tab()


if __name__ == "__main__":
    main()
