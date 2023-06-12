# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:31:56 2023

@author: ManosIeronymakisProb
"""

import pandas as pd
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\ELU\\M5 - W4 Assignment Unsupervised Learning")
filepath =  "train.csv"
df = pd.read_csv(filepath)

# Check for missing values
print(df.isnull().sum())

# Check for duplicates
print(df.duplicated().sum())

object_columns = df.select_dtypes(include=['object']).columns

# Get dummy variables for object columns
df_dummies = pd.get_dummies(df, columns=object_columns)
df_dummies = df_dummies.astype(int)

# Print the shape of the new dataframe
print(df_dummies.shape)

def get_duplicate_columns(df_dummies):
    duplicate_column_names = set()
    for x in range(df.shape[1]):
        col = df.iloc[:, x]
        for y in range(x + 1, df.shape[1]):
            other_col = df.iloc[:, y]
            if col.equals(other_col):
                duplicate_column_names.add(df.columns.values[y])
    return list(duplicate_column_names)

duplicate_columns = get_duplicate_columns(df_dummies)
print(f'Duplicate Columns are as follows: {duplicate_columns}')

# Remove 'ID' and 'y' columns
df_features = df_dummies.drop(['ID', 'y'], axis=1)

# Remove duplicate columns
df_features = df_features.drop(duplicate_columns, axis=1)

# Normalize the features
scaler = StandardScaler()
df_features_normalized = scaler.fit_transform(df_features)

# Perform PCA with all components
pca = PCA()
pca.fit(df_features_normalized)

# Calculate cumulative explained variance
cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Find the number of components for desired explained variance
desired_variance = 0.95  # Change this to your desired explained variance ratio
n_components = np.where(cumulative_explained_variance > desired_variance)[0][0] + 1  # Adding 1 because of zero-indexing

print(f'The optimal number of components is: {n_components}')

# Plot the cumulative explained variance
plt.figure(figsize=(10, 7))
plt.plot(range(1, len(cumulative_explained_variance)+1), cumulative_explained_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance vs Number of Components')
plt.grid(True)
plt.show()


# Perform PCA with the optimal number of components
pca = PCA(n_components=247)
df_pca = pca.fit_transform(df_features_normalized)

# Calculate the absolute correlation between original features and PCA components
correlations = pd.DataFrame(columns=['Feature', 'PC', 'Correlation'])
for i in range(df_features_normalized.shape[1]):
    feature = df_features.columns[i]
    for j in range(n_components):
        correlation = np.abs(np.corrcoef(df_features_normalized[:, i], df_pca[:, j])[0, 1])
        correlations = pd.concat([correlations, pd.DataFrame({'Feature': [feature], 'PC': [f'PC{j+1}'], 'Correlation': [correlation]})], ignore_index=True)

# Select features with absolute correlation of at least 0.75 with any of the PCA components
selected_features = correlations[correlations['Correlation'] >= 0.75]

# Print the number of selected features and the selected features
print(f"The number of selected features is: {len(selected_features)}")
print("Selected features:")
for feature, pc, correlation in selected_features.values:
    print(f"Feature: {feature}, PC: {pc}, Correlation: {correlation}")







