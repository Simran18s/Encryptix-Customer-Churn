# Encryptix-Customer-Churn

# Customer Churn Prediction

This project aims to develop a machine learning model to predict customer churn for a subscription-based service or business. Using historical customer data, including features like usage behavior and customer demographics, we apply algorithms such as Logistic Regression, Random Forests, and Gradient Boosting to predict churn.

## Table of Contents
-Project Overview
-Features
-Installation

## Project Overview
Customer churn prediction is crucial for businesses to retain customers and reduce revenue loss. By analyzing historical data, we can identify patterns that indicate whether a customer is likely to churn. This project uses a combination of demographic information and usage behavior to train predictive models.

## Features
  - Data Preprocessing: Handling missing values, feature engineering, and normalization.
  - Exploratory Data Analysis (EDA): Visualizing data distributions and relationships.
  - Model Training: Implementing Logistic Regression, Random Forests, and Gradient Boosting.
  - Model Evaluation: Assessing model performance using metrics like accuracy, precision, recall, and F1-score.
  - Streamlit Web App: Interactive interface to visualize data and predict customer churn.

### Create a virtual environment:

    python -m venv venv

    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

### Install the required packages:

    pip install -r requirements.txt

### Run the Streamlit application:

    streamlit run app.py
    
