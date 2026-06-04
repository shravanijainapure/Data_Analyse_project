# 📊 Enterprise Workforce Analytics & Attrition Dashboard

An end-to-end data analytics and predictive machine learning web application built to help Human Resources departments identify, analyze, and mitigate employee turnover risks.

## 🚀 Live Demo
👉 **[Link your deployed Streamlit App here if you host it online, otherwise delete this line]**

## 🧠 Project Architecture & Workflow
The application functions through a two-stage pipeline:
1. **Model Training (`train_model.py`):** Cleans raw HR data, handles categorical feature encoding, separates target matrices, and trains a Scikit-Learn `Random Forest Classifier` to predict attrition trends.
2. **Interactive UI (`app.py`):** A Streamlit interface that hosts dynamic key performance indicator (KPI) metric modules, interactive Plotly visualizations, and an active predictive risk simulator.

## 🛠️ Technology Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Data Visualization:** Plotly Express
* **Machine Learning:** Scikit-Learn
* **Application Framework:** Streamlit

## 📦 Local Installation & Setup

1. Clone the repository to your local machine:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/corporate-workforce-analytics.git](https://github.com/YOUR_USERNAME/corporate-workforce-analytics.git)
