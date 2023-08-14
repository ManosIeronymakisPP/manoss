# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:20:05 2023

@author: ManosIeronymakisProb
"""

import seaborn as sns

titanic_df = sns.load_dataset('titanic')

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier

# Prepare the data for modeling
titanic_df.drop(['alive', 'embark_town', 'who', 'adult_male', 'deck', 'embarked', 'alone'], axis=1, inplace=True)
titanic_df['age'].fillna(titanic_df['age'].median(), inplace=True)
titanic_df['fare'].fillna(titanic_df['fare'].median(), inplace=True)
titanic_df = pd.get_dummies(titanic_df, columns=['sex', 'class'], drop_first=True)

# Split the data into features (X) and target variable (y)
X = titanic_df.drop('survived', axis=1)
y = titanic_df['survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for the grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Perform grid search cross-validation
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Refit the best model on the whole dataset
best_model = grid_search.best_estimator_
best_model.fit(X, y)



import shap

# Initialize the Tree SHAP explainer
explainer = shap.TreeExplainer(best_model)

# Compute SHAP values
shap_values = explainer.shap_values(X_test)

# Plot summary plot of SHAP values
shap.summary_plot(shap_values, X_test)



import pandas as pd
import random
import transformers
import shap
import os


# Download the IMDb test dataset and select 20 samples
os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\ELU\\M6 - W7 Assignment Explain ML predictions using SHAP")
filepath = "imdb_top_1000.csv"
imdb_test_df = pd.read_csv(filepath)
random.seed(42)  # For reproducibility
selected_indices = random.sample(range(len(imdb_test_df)), 20)
selected_texts = imdb_test_df.iloc[selected_indices]['Overview']

# Convert the pandas Series to a list of strings
selected_texts = selected_texts.tolist()

# Download the pre-trained DistilBERT model for sentiment analysis
tokenizer = transformers.DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = transformers.DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Define a function for making predictions using the DistilBERT model
def predict_sentiment(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = logits.softmax(dim=1)
    sentiment_labels = ["negative", "positive"]
    predicted_labels = [sentiment_labels[pred.item()] for pred in probabilities.argmax(dim=1)]
    return predicted_labels

# Make predictions on the selected texts
predicted_labels = predict_sentiment(selected_texts)

# Print the selected texts and their corresponding predicted labels
for text, label in zip(selected_texts, predicted_labels):
    print("Text:", text)
    print("Predicted Sentiment:", label)
    print()

# Select two texts that were correctly predicted (either both positive or both negative)
correctly_predicted_texts = [text for text, label in zip(selected_texts, predicted_labels) if label == "positive" or label == "negative"]
if len(correctly_predicted_texts) >= 2:
    text_1, text_2 = correctly_predicted_texts[:2]

    # Create a SHAP explainer for text classification
    explainer = shap.Explainer(model, tokenizer)

    # Get the SHAP values for the two selected texts
    shap_values_1 = explainer(text_1)
    shap_values_2 = explainer(text_2)

    # Analyze the SHAP results to understand word contributions
    print("SHAP Values for Text 1:")
    print(shap_values_1)

    print("\nSHAP Values for Text 2:")
    print(shap_values_2)

else:
    print("Not enough correctly predicted texts to continue.")
























