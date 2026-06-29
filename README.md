# Telecom Customer Churn Prediction

This project is a machine learning web app for **telecom subscription retention**. It predicts whether a customer is likely to churn and gives a simple risk band so the business can decide who needs follow-up.

The project has been rebuilt to be easier to understand and easier to use:

- Clear niche: telecom subscription retention
- Simple input form with only 7 practical factors
- Better UI with clear navigation
- Single customer prediction
- Batch CSV prediction with a downloadable template
- Risk pattern explorer for explanation
- Scikit-learn pipeline with preprocessing and model saved together

## Why This Project

Telecom companies lose revenue when customers cancel plans. A churn model helps identify customers who may leave soon, so the company can offer support, payment help, plan changes, or retention offers.

## Features Used

The model uses only these important fields:

```text
Tenure
Usage Frequency
Support Calls
Payment Delay
Subscription Type
Contract Length
Total Spend
```

These are easier to collect and explain than a long list of customer details.

## Project Structure

```text
telecom customer churn prediction/
|
|-- app/
|   |-- app.py              # Streamlit web application
|   `-- predict.py          # Prediction helper functions
|
|-- Data/
|   `-- archive/
|       `-- customer_churn_dataset-testing-master.csv
|
|-- models/
|   |-- churn_pipeline.pkl  # Trained model pipeline
|   `-- metrics.json        # Model evaluation scores
|
|-- src/
|   |-- config.py
|   |-- data_loader.py
|   |-- data_cleaning.py
|   |-- preprocessing.py
|   |-- model.py
|   |-- evaluation.py
|   `-- save_model.py
|
|-- train.py
|-- requirements.txt
|-- README.md
`-- .streamlit/config.toml
```

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest Classifier
- Streamlit
- Plotly
- Joblib

## Install Requirements

```powershell
pip install -r requirements.txt
```

## Train the Model

```powershell
python train.py
```

This creates:

```text
models/churn_pipeline.pkl
models/metrics.json
```

## Run the Web App

```powershell
streamlit run app/app.py
```

## App Pages

### Overview

Shows dataset size, churn rate, model scores, and simple charts.

### Predict One Customer

Enter seven fields and get:

- Churn probability
- Prediction: Churn or Stay
- Risk band: Low, Watch, or High
- Suggested retention action

### Batch CSV Scoring

Upload a CSV file and download the scored results. A CSV template is available inside the app.

### Risk Patterns

Explore which factors are linked with churn, such as payment delay, support calls, usage frequency, and tenure.

## Deploy on Streamlit Cloud

Push the project to GitHub, then deploy it on Streamlit Cloud.

Use this as the main file path:

```text
app/app.py
```

## Resume Summary

Built a telecom customer churn prediction web app using Python, Scikit-learn, and Streamlit. Developed a full ML pipeline for preprocessing, training, evaluation, and deployment. Added single and batch prediction workflows with churn probability, risk bands, and interactive risk analysis.
