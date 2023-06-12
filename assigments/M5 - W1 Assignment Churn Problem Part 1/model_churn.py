# -*- coding: utf-8 -*-
"""
Created on Fri May  5 16:19:52 2023

@author: ManosIeronymakisProb
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\ELU\\M5 - W1 Assignment Churn Problem Part 1")
filepath =  "WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(filepath)




def clean_data(df):
    # Convert TotalCharges to a numeric data type
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
    # Remove customerID column
    df.drop('customerID', axis=1, inplace=True)
        
    # Replace spaces with underscores in column names
    df.columns = df.columns.str.replace(' ', '_')
        
    # Replace 'No internet service' with 'No' for some columns
    replace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in replace_cols:
        df[col] = df[col].replace({'No internet service': 'No'})
        
    # Replace 'No phone service' with 'No' for MultipleLines column
    df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})
        
    # Map values to 0 and 1 for SeniorCitizen column
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        
    # Create dummy variables for categorical columns
    cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        
    # Replace missing values in TotalCharges column with median
    median = df['TotalCharges'].median()
    df['TotalCharges'].fillna(median, inplace=True)
        
    return df_cleaned

df_cleaned= clean_data(df)


# Define the process for each feature
process_dict = {
    'gender': 'The gender column currently holds the values “Male” and “Female”. To prepare for future observations, we can map the values "M" to "Male" and "F" to "Female".',
    'SeniorCitizen': 'The SeniorCitizen column currently holds the values 0 and 1. To prepare for future observations, we can map the value 2 to "Unknown".',
    'Partner': 'The Partner column currently holds the values "Yes" and "No". No additional cleaning is necessary.',
    'Dependents': 'The Dependents column currently holds the values "Yes" and "No". No additional cleaning is necessary.',
    'tenure': 'The tenure column represents the number of months the customer has been with the company. No additional cleaning is necessary.',
    'PhoneService': 'The PhoneService column currently holds the values "Yes" and "No". No additional cleaning is necessary.',
    'MultipleLines': 'The MultipleLines column currently holds the values "Yes", "No", and "No phone service". To prepare for future observations, we can map the value "N/A" to "No phone service".',
    'InternetService': 'The InternetService column currently holds the values ',
    'OnlineSecurity': 'The OnlineSecurity column currently holds the values "Yes", "No", and "No internet service". To prepare for future observations, we can map the value "N/A" to "No internet service".',
    'OnlineBackup': 'The OnlineBackup column currently holds the values "Yes", "No", and "No internet service". To prepare for future observations, we can map the value "N/A" to "No internet service".',
    'DeviceProtection': 'The DeviceProtection column currently holds the values "Yes", "No", and "No internet service". To prepare for future observations, we can map the value "N/A" to "No internet service".',
    'TechSupport': 'The TechSupport column currently holds the values "Yes", "No", and "No internet service". To prepare for future observations, we can map the value "N/A" to "No internet service".',
    'StreamingTV': 'The StreamingTV column currently holds the values "Yes", "No", and "No internet service". To prepare for future observations, we can map the value "N/A" to "No internet service".',
    'StreamingMovies': 'The StreamingMovies column currently holds the values "Yes", "No", and "No internet service". To prepare for future observations, we can map the value "N/A" to "No internet service".',
    'Contract': 'The Contract column currently holds the values "Month-to-month", "One year", and "Two year". No additional cleaning is necessary.',
    'PaperlessBilling': 'The PaperlessBilling column currently holds the values "Yes" and "No". No additional cleaning is necessary.',
    'PaymentMethod': 'The PaymentMethod column currently holds the values "Electronic check", "Mailed check", "Bank transfer (automatic)", and "Credit card (automatic)". No additional cleaning is necessary.',
    'MonthlyCharges': 'The MonthlyCharges column represents the amount charged to the customer each month. No additional cleaning is necessary.',
    'TotalCharges': 'The TotalCharges column represents the total amount charged to the customer over their tenure. No additional cleaning is necessary.'
}


# Print out the processes for each feature
for col, process in process_dict.items():
    print(f'{col}: {process}')


# Create a new column 'Churn_Yes' based on the 'Churn' column
df_cleaned['Churn_Yes'] = df_cleaned['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Drop the original 'Churn' column
df_cleaned.drop('Churn', axis=1, inplace=True)

# Split the data into features and target
X = df_cleaned.drop('Churn_Yes', axis=1)
y = df_cleaned['Churn_Yes']

# Initialize the logistic regression model
lr = LogisticRegression()

# Perform cross-validation with 5 folds and accuracy score as evaluation metric
cv_scores = cross_val_score(lr, X, y, cv=5, scoring='accuracy')

# Print the average score and standard deviation
print('Average accuracy:', cv_scores.mean())
print('Standard deviation:', cv_scores.std())











