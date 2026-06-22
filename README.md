# 🏠 AI-Based House Price Prediction Model

## 📌 Overview

This project is an end-to-end Machine Learning application that predicts residential property prices using housing and geographical features. The system leverages advanced regression and ensemble learning algorithms to provide accurate house price estimates while offering explainable AI insights for transparency.

The application is deployed as an interactive web dashboard using Streamlit, enabling users to input property details and receive instant price predictions along with model explanations and performance analytics.

---

## 🚀 Features

### 🔹 House Price Prediction

- Predicts property prices based on housing attributes and location data.
- Supports real-time inference through an interactive web interface.

### 🔹 Multiple Machine Learning Models

- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

### 🔹 Explainable AI

- SHAP-based feature importance visualization.
- Understand factors influencing each prediction.

### 🔹 Data Analytics Dashboard

- Dataset exploration
- Feature distributions
- Correlation analysis
- Model performance comparison

### 🔹 Confidence Estimation

- Provides confidence intervals for predictions.

### 🔹 Geospatial Insights

- Uses latitude and longitude information for enhanced prediction accuracy.

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Machine Learning

- Scikit-Learn
- XGBoost
- SHAP

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn
- Plotly

### Deployment

- Streamlit

---

## 📊 Dataset

Dataset: California Housing Dataset

Features:

- Median Income
- House Age
- Average Rooms
- Average Bedrooms
- Population
- Occupancy
- Latitude
- Longitude

Target:

- Median House Value

---

## ⚙️ Machine Learning Pipeline

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Training
6. Hyperparameter Tuning
7. Model Evaluation
8. Explainability Analysis
9. Deployment

---

## 📈 Model Performance

| Model             | Evaluation         |
| ----------------- | ------------------ |
| Ridge Regression  | Baseline           |
| Lasso Regression  | Baseline           |
| Random Forest     | High Accuracy      |
| Gradient Boosting | Strong Performance |
| XGBoost           | Best Performance   |

Best Model: XGBoost

Evaluation Metrics:

- R² Score
- MAE
- RMSE
- MAPE
- Cross Validation Score

---

## 🖥️ Running Locally

### Clone Repository

```bash
git clone https://github.com/yourusername/house-price-prediction-ai.git
cd house-price-prediction-ai
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
house-price-prediction-ai/
│
├── app.py
├── requirements.txt
├── data/
├── models/
├── notebooks/
├── assets/
└── README.md
```

---

## 🎯 Business Applications

- Real Estate Price Estimation
- Property Investment Analysis
- Market Trend Evaluation
- Smart Property Recommendation Systems
- Real Estate Decision Support Systems

---

## 🔮 Future Enhancements

- Deep Learning Models
- Live Real Estate API Integration
- City-wise Price Prediction
- Property Recommendation Engine
- Interactive Maps Dashboard
- Market Trend Forecasting

---

## 👨‍💻 Author

Vivek M

B.Tech CSE (Artificial Intelligence & Machine Learning)

VIT-AP University

LinkedIn: https://www.linkedin.com/in/vivek-m-514682332/

GitHub: https://github.com/vivekm221006
