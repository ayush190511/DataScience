# 🎓 Student Math Score Prediction (End-to-End ML + MLOps)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end **Machine Learning and MLOps-style** project for
predicting **student math performance** using multiple regression
models, experiment tracking with **MLflow + DagsHub**, and deployment
via **Streamlit**.

------------------------------------------------------------------------

## 🚀 Project Overview

This project builds a regression system that predicts **Math Score**
using demographic and academic features.\
It demonstrates real-world ML engineering practices including:

-   Modular ML pipeline design\
-   Multi-model benchmarking\
-   Hyperparameter tuning\
-   Experiment tracking with MLflow\
-   SHAP-based interpretability\
-   Production-ready Streamlit deployment

------------------------------------------------------------------------

## 🧠 Key Features

-   ✅ Modular ML pipeline architecture\
-   ✅ Comparison of 8 regression algorithms\
-   ✅ Hyperparameter tuning\
-   ✅ MLflow experiment tracking\
-   ✅ DagsHub integration\
-   ✅ SHAP model interpretability\
-   ✅ Training--serving consistency via full sklearn Pipeline\
-   ✅ Streamlit web app for real-time predictions

------------------------------------------------------------------------

## 🏗 Project Workflow

Data Ingestion\
↓\
Data Transformation (ColumnTransformer)\
↓\
Model Benchmarking & Tuning\
↓\
Best Model Selection\
↓\
MLflow Logging\
↓\
SHAP Interpretability\
↓\
Streamlit Deployment

------------------------------------------------------------------------

## 📂 Project Structure

    ├── artifacts/
    │   ├── model.pkl
    │   ├── model_comparison.csv
    │   └── shap_summary.png
    │
    ├── src/DataScienceProject/
    │   ├── components/
    │   │   ├── data_ingestion.py
    │   │   ├── data_transformation.py
    │   │   └── model_trainer.py
    │   ├── utils.py
    │   ├── logger.py
    │   └── exception.py
    │
    ├── streamlit_app.py
    ├── main.py
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 📊 Dataset

Based on the **Students Performance Dataset**.

### Target Variable

`math_score`

------------------------------------------------------------------------

## 🤖 Models Evaluated

-   Linear Regression\
-   K-Nearest Neighbors\
-   Decision Tree\
-   Random Forest\
-   Gradient Boosting\
-   AdaBoost\
-   XGBoost\
-   CatBoost

Best model selected based on **R² Score**.

------------------------------------------------------------------------

## 📈 Evaluation Metrics

-   RMSE\
-   MAE\
-   R² Score

------------------------------------------------------------------------

## 🌐 Streamlit Deployment

Run locally with:

``` bash
streamlit run streamlit_app.py
```

Open browser at:

http://localhost:8501

<p align="center">
  <a href="https://studentmarksdsproject.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Live-Demo-brightgreen?style=for-the-badge" />
  </a>
</p>

------------------------------------------------------------------------

## ⚙️ Tech Stack

-   Python\
-   scikit-learn\
-   XGBoost\
-   CatBoost\
-   MLflow\
-   DagsHub\
-   SHAP\
-   Streamlit\
-   Pandas\
-   NumPy

------------------------------------------------------------------------

## 👨‍💻 Author

**Ayush Mishra**\

------------------------------------------------------------------------

⭐ If you found this project useful, consider giving it a star!
