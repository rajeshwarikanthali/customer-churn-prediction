import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from app.predict import (
        FEATURE_COLUMNS,
        load_metrics,
        load_model,
        load_reference_data,
        predict_churn,
        risk_band,
    )
except ModuleNotFoundError:
    from predict import (
        FEATURE_COLUMNS,
        load_metrics,
        load_model,
        load_reference_data,
        predict_churn,
        risk_band,
    )


st.set_page_config(
    page_title="Telecom Retention Desk",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background: #f6f7f5;
        color: #1d2528;
    }

    .main h1, .main h2, .main h3, .main p, .main label, .main span, .main div {
        color: #1d2528;
    }

    [data-testid="stSidebar"] {
        background: #172522;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #f4f7f5;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        padding: .35rem .5rem;
        margin-bottom: .35rem;
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #1d2528;
    }

    .top-band {
        background: #ffffff;
        border: 1px solid #dfe5df;
        border-left: 7px solid #136f63;
        border-radius: 8px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 16px 34px rgba(29, 37, 40, 0.07);
    }

    .eyebrow {
        color: #136f63;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .45rem;
    }

    .top-band h1 {
        margin: 0 0 .3rem 0;
        font-size: 2.15rem;
        line-height: 1.15;
        letter-spacing: 0;
    }

    .top-band p {
        color: #5f6b66;
        margin: 0;
        max-width: 820px;
    }

    .card {
        background: #ffffff;
        border: 1px solid #dfe5df;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 12px 28px rgba(29, 37, 40, 0.06);
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #dfe5df;
        border-radius: 8px;
        padding: 1rem;
        min-height: 108px;
        box-shadow: 0 12px 28px rgba(29, 37, 40, 0.06);
    }

    .metric-label {
        color: #65736c;
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-weight: 800;
    }

    .metric-value {
        color: #1d2528;
        font-size: 1.7rem;
        font-weight: 800;
        margin-top: .25rem;
    }

    .risk-box {
        border-radius: 8px;
        padding: 1rem 1.1rem;
        background: #ffffff;
        border: 1px solid #dfe5df;
        border-left: 7px solid #136f63;
        box-shadow: 0 12px 28px rgba(29, 37, 40, 0.06);
    }

    .risk-box.high {
        border-left-color: #b42318;
    }

    .risk-box.watch {
        border-left-color: #b7791f;
    }

    .risk-title {
        font-size: 1.15rem;
        font-weight: 800;
        margin-bottom: .25rem;
    }

    .risk-copy {
        color: #5f6b66;
        margin: 0;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        background: #136f63;
        border: 1px solid #136f63;
        border-radius: 8px;
        color: white;
        font-weight: 800;
        min-height: 2.65rem;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background: #0f5d53;
        border-color: #0f5d53;
        color: white;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    input,
    textarea {
        background: #ffffff;
        color: #1d2528;
        border-radius: 8px;
    }

    [data-testid="stDataFrame"] * {
        color: #1d2528;
    }
</style>
"""


@st.cache_resource(show_spinner=False)
def cached_model():
    return load_model()


@st.cache_data(show_spinner=False)
def cached_reference_data():
    return load_reference_data()


@st.cache_data(show_spinner=False)
def cached_metrics():
    return load_metrics()


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def app_header(title, description):
    st.markdown(
        f"""
        <div class="top-band">
            <div class="eyebrow">Telecom subscription retention</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def single_customer_input():
    left, middle, right = st.columns(3)

    with left:
        tenure = st.slider("Months with company", 1, 72, 18)
        usage = st.slider("Monthly usage frequency", 1, 35, 15)
        total_spend = st.number_input("Total spend", min_value=0.0, value=650.0, step=25.0)

    with middle:
        support_calls = st.slider("Support calls", 0, 10, 2)
        payment_delay = st.slider("Payment delay in days", 0, 30, 5)

    with right:
        subscription = st.selectbox("Subscription type", ["Basic", "Standard", "Premium"])
        contract = st.selectbox("Contract length", ["Monthly", "Quarterly", "Annual"])

    return pd.DataFrame(
        [
            {
                "Tenure": tenure,
                "Usage Frequency": usage,
                "Support Calls": support_calls,
                "Payment Delay": payment_delay,
                "Subscription Type": subscription,
                "Contract Length": contract,
                "Total Spend": total_spend,
            }
        ]
    )


def show_prediction(probability, prediction):
    band = risk_band(probability)
    css_class = "high" if band == "High" else "watch" if band == "Watch" else "low"
    title = {
        "High": "High churn risk",
        "Watch": "Needs attention",
        "Low": "Low churn risk",
    }[band]
    message = {
        "High": "Prioritize this customer for a retention call, payment help, or plan review.",
        "Watch": "Monitor this customer and consider a light retention offer.",
        "Low": "This customer currently appears stable.",
    }[band]

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Churn probability", f"{probability:.1%}")
    with col2:
        metric_card("Prediction", "Churn" if prediction == 1 else "Stay")
    with col3:
        metric_card("Risk band", band)

    st.progress(probability)
    st.markdown(
        f"""
        <div class="risk-box {css_class}">
            <div class="risk-title">{title}</div>
            <p class="risk-copy">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_batch(model, batch_data):
    missing = [column for column in FEATURE_COLUMNS if column not in batch_data.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    predictions, probabilities = predict_churn(model, batch_data[FEATURE_COLUMNS])
    result = batch_data.copy()
    result["Churn Probability"] = probabilities
    result["Prediction"] = ["Churn" if value == 1 else "Stay" for value in predictions]
    result["Risk Band"] = [risk_band(float(value)) for value in probabilities]
    return result


def overview_page(data, metrics):
    app_header(
        "Retention Desk",
        "A focused churn prediction app for telecom subscriptions. It uses practical customer behavior signals instead of asking for too many hard-to-fill details.",
    )

    churn_rate = data["Churn"].mean()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Customers", f"{len(data):,}")
    with col2:
        metric_card("Churn rate", f"{churn_rate:.1%}")
    with col3:
        metric_card("Model F1", f"{metrics.get('f1_score', 0):.2f}")
    with col4:
        metric_card("ROC AUC", f"{metrics.get('roc_auc', 0):.2f}")

    left, right = st.columns(2)
    with left:
        fig = px.histogram(
            data,
            x="Payment Delay",
            color="Churn",
            barmode="overlay",
            color_discrete_sequence=["#136f63", "#b42318"],
        )
        fig.update_layout(title="Payment delay pattern", margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        contract = data.groupby("Contract Length", as_index=False)["Churn"].mean()
        fig = px.bar(
            contract,
            x="Contract Length",
            y="Churn",
            color="Contract Length",
            color_discrete_sequence=["#136f63", "#d6a84f", "#7a4e2d"],
        )
        fig.update_layout(title="Churn rate by contract", margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def predict_page(model):
    app_header(
        "Single Customer Prediction",
        "Enter seven easy customer signals to estimate churn risk and get a simple retention recommendation.",
    )

    customer = single_customer_input()

    if st.button("Predict customer risk", use_container_width=True):
        predictions, probabilities = predict_churn(model, customer)
        show_prediction(float(probabilities[0]), int(predictions[0]))

        with st.expander("View values sent to the model"):
            st.dataframe(customer, use_container_width=True, hide_index=True)


def batch_page(model):
    app_header(
        "Batch Prediction",
        "Upload a CSV with the same seven input columns and download scored churn results.",
    )

    template = pd.DataFrame(
        [
            {
                "Tenure": 18,
                "Usage Frequency": 15,
                "Support Calls": 2,
                "Payment Delay": 5,
                "Subscription Type": "Standard",
                "Contract Length": "Monthly",
                "Total Spend": 650,
            }
        ]
    )

    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        "churn_batch_template.csv",
        "text/csv",
        use_container_width=True,
    )

    uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Use the template above if you are unsure about the exact column names.")
        return

    try:
        batch_data = pd.read_csv(uploaded_file)
        result = score_batch(model, batch_data)

        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Rows scored", f"{len(result):,}")
        with col2:
            metric_card("Average risk", f"{result['Churn Probability'].mean():.1%}")
        with col3:
            metric_card("High risk", f"{(result['Risk Band'] == 'High').sum():,}")

        fig = px.histogram(
            result,
            x="Churn Probability",
            color="Risk Band",
            nbins=20,
            color_discrete_map={"Low": "#136f63", "Watch": "#d6a84f", "High": "#b42318"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(result.sort_values("Churn Probability", ascending=False), use_container_width=True, hide_index=True)
        st.download_button(
            "Download scored results",
            result.to_csv(index=False).encode("utf-8"),
            "churn_predictions.csv",
            "text/csv",
            use_container_width=True,
        )
    except Exception as error:
        st.error(str(error))


def risk_page(data):
    app_header(
        "Risk Pattern Explorer",
        "Use this page to understand which customer patterns usually need retention attention.",
    )

    selected = st.selectbox(
        "Choose a risk factor",
        ["Support Calls", "Payment Delay", "Usage Frequency", "Tenure", "Total Spend"],
    )

    fig = px.box(
        data,
        x="Churn",
        y=selected,
        color="Churn",
        color_discrete_sequence=["#136f63", "#b42318"],
    )
    fig.update_layout(margin=dict(l=10, r=10, t=35, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="card">
            <strong>How to read this:</strong>
            Customers with many support calls, longer payment delays, low usage, or short tenure often need earlier follow-up.
            This page is for explanation, while the prediction page is for scoring one customer.
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
model = cached_model()
reference_data = cached_reference_data()
metrics = cached_metrics()

with st.sidebar:
    st.title("Retention Desk")
    st.caption("Telecom churn project")
    page = st.radio(
        "Choose what you want to do",
        ["Overview", "Predict one customer", "Batch CSV scoring", "Risk patterns"],
    )
    st.divider()
    st.caption("Model ready")
    st.caption("Inputs: 7 practical fields")

if page == "Overview":
    overview_page(reference_data, metrics)
elif page == "Predict one customer":
    predict_page(model)
elif page == "Batch CSV scoring":
    batch_page(model)
else:
    risk_page(reference_data)
