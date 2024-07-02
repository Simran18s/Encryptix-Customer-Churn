import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import streamlit as st

st.header("Customer Churn Predictor")
# Load the dataset
data = pd.read_csv('Churn_Modelling.csv')

# Display basic information
st.write("Dataset Information:")
st.write(data.info())
st.write(data.describe())
st.write(data.head())

# Check for missing values
st.write("Missing Values:")
st.write(data.isnull().sum())

# Exploratory Data Analysis
sns.countplot(x='Exited', data=data)
plt.title('Churn Distribution')
st.pyplot(plt)

# Drop unnecessary columns
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Encode categorical features
for column in ['Geography', 'Gender']:
    data[column] = data[column].astype(str)  # Ensure the column is of type string
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])

# Split data into features and target variable
X = data.drop(['Exited'], axis=1)
y = data['Exited']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Train Random Forest model
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

# Train Gradient Boosting model
gb_clf = GradientBoostingClassifier()
gb_clf.fit(X_train, y_train)

# Evaluate Logistic Regression
log_reg_pred = log_reg.predict(X_test)
st.write("Logistic Regression")
st.write(classification_report(y_test, log_reg_pred))
st.write("ROC-AUC Score:", roc_auc_score(y_test, log_reg_pred))

# Evaluate Random Forest
rf_pred = rf_clf.predict(X_test)
st.write("Random Forest")
st.write(classification_report(y_test, rf_pred))
st.write("ROC-AUC Score:", roc_auc_score(y_test, rf_pred))

# Evaluate Gradient Boosting
gb_pred = gb_clf.predict(X_test)
st.write("Gradient Boosting")
st.write(classification_report(y_test, gb_pred))
st.write("ROC-AUC Score:", roc_auc_score(y_test, gb_pred))

# Streamlit app
st.title('Bank Customer Churn Prediction')

# User inputs for model
st.sidebar.header('Customer Input Features')
def user_input_features():
    CreditScore = st.sidebar.slider('Credit Score', 350, 850, 500)
    Geography = st.sidebar.selectbox('Geography', ('France', 'Spain', 'Germany'))
    Gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    Age = st.sidebar.slider('Age', 18, 92, 30)
    Tenure = st.sidebar.slider('Tenure', 0, 10, 5)
    Balance = st.sidebar.slider('Balance', 0, 250000, 50000)
    NumOfProducts = st.sidebar.slider('Number of Products', 1, 4, 1)
    HasCrCard = st.sidebar.selectbox('Has Credit Card?', (0, 1))
    IsActiveMember = st.sidebar.selectbox('Is Active Member?', (0, 1))
    EstimatedSalary = st.sidebar.slider('Estimated Salary', 0, 200000, 50000)
    
    features = {'CreditScore': CreditScore,
                'Geography': Geography,
                'Gender': Gender,
                'Age': Age,
                'Tenure': Tenure,
                'Balance': Balance,
                'NumOfProducts': NumOfProducts,
                'HasCrCard': HasCrCard,
                'IsActiveMember': IsActiveMember,
                'EstimatedSalary': EstimatedSalary}
    input_data = pd.DataFrame(features, index=[0])
    return input_data

input_df = user_input_features()

# Ensure all columns are of the appropriate data type
input_df['Geography'] = input_df['Geography'].astype(str)
input_df['Gender'] = input_df['Gender'].astype(str)

# Combine user input features with entire dataset (this is needed for scaling)
full_data = pd.concat([input_df, data.drop(columns=['Exited'])], axis=0)

# Encode categorical features
for column in ['Geography', 'Gender']:
    full_data[column] = full_data[column].astype(str)  # Ensure the column is of type string
    le = LabelEncoder()
    full_data[column] = le.fit_transform(full_data[column])

scaled_input_df = scaler.transform(full_data[:1])

# Prediction using the best model (Assuming Gradient Boosting was the best)
prediction = gb_clf.predict(scaled_input_df)
prediction_proba = gb_clf.predict_proba(scaled_input_df)

st.subheader('Prediction')
st.write('Churn' if prediction == 1 else 'No Churn')

st.subheader('Prediction Probability')
st.write(prediction_proba)
