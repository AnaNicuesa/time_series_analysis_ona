# Retail Sales Forecasting Project

## Project Overview

This project explores multiple forecasting approaches for predicting daily retail sales using time series data.

The workflow combines:

* exploratory data analysis,
* feature engineering,
* classical statistical forecasting,
* machine learning,
* deep learning,
* experiment tracking,
* and interactive deployment.

The main objective was to compare forecasting performance across different modelling strategies while building a reproducible end-to-end forecasting pipeline.

---

# Business Problem

Retail companies rely heavily on accurate demand forecasts.

Poor forecasting performance can lead to:

* stock shortages,
* excess inventory,
* inefficient operations,
* and financial losses.

The goal of this project was to forecast future daily unit sales using historical retail data and external variables.

---

# Dataset

The project uses retail sales time series data enriched with:

* calendar variables,
* holiday indicators,
* rolling statistics,
* lag features,
* and oil price information.

The final dataset includes:

* daily unit sales,
* engineered temporal features,
* and external signals.

---

# Exploratory Data Analysis

During exploratory analysis and data preparation, several important issues and business patterns were identified:

- missing values existed in some external variables,
- sales behavior was highly volatile,
- weekends consistently showed higher demand,
- sudden spikes and drops appeared frequently,
- and demand patterns evolved over time.

Missing values were carefully handled during preprocessing to maintain temporal consistency and avoid introducing leakage into the forecasting workflow.

These observations motivated the creation of engineered features for the machine learning models.


---

# Feature Engineering

Feature engineering became one of the most important parts of the project.

The following features were created:

## Temporal Features

* month
* weekday
* weekend indicators
* year

## Lag Features

* lag_1
* lag_7
* lag_14
* lag_30

## Rolling Statistics

* rolling averages
* rolling standard deviations

## External Features

* holiday indicators
* oil prices
* rolling oil statistics

---

# Forecasting Models

The project compares several forecasting approaches.

## Statistical Models

* ARIMA
* Exponential Smoothing
* Prophet

## Machine Learning

* XGBoost

## Deep Learning

* LSTM Neural Network

---

# Model Evaluation

Models were evaluated using:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* MAPE (Mean Absolute Percentage Error)
* Bias
* R-squared

The experiments showed that XGBoost achieved the strongest balance between forecasting accuracy and stability.

---

# MLflow Integration

MLflow was used for:

* experiment tracking,
* metric comparison,
* parameter logging,
* and model management.

This allowed the forecasting experiments to become more reproducible and organized.

---

# Streamlit Application

A Streamlit application was created to provide:

* interactive forecast visualization,
* model exploration,
* and simplified forecasting workflows.

---

# Main Notebooks

* `W3-mlflow.ipynb` → MLflow experimentation, feature engineering, model comparison, and best model selection
* `W3-streamlit.ipynb` → Streamlit application setup and forecasting interface


# Repository Structure

```text
time_series_analysis_ona/
│
├── app/
├── data/
├── models/
├── notebooks/
├── outputs/
├── README.md
├── requirements.txt
├── requirements_clean.txt
└── .gitignore
```

---

# Main Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Prophet
* PyTorch
* MLflow
* Streamlit
* Matplotlib

---

# Key Lessons Learned

This project demonstrated that forecasting is not only about selecting a model.

Understanding the data and engineering meaningful temporal features were critical for improving forecasting performance.

The project also highlighted the strengths and limitations of:

* statistical forecasting,
* machine learning,
* and deep learning approaches.

---

# Author

Ana Nicuesa

