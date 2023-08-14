# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:04:02 2023

@author: ManosIeronymakisProb
"""

import bz2
import re
import os
import pandas as pd

os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\ELU\M6- W3 Assignment NLP")

train_file = bz2.BZ2File("train.ft.txt.bz2")

# Load and decode
lines = [x.decode('utf-8') for x in train_file.readlines()]

# Extract reviews and labels using regular expressions and named groups
score_review_list = [re.match(r"__label__(?P<score>\d+) (?P<review>.*)", l).groupdict() for l in lines]


df = pd.DataFrame(score_review_list, columns=['score', 'review'])

#############

df['n_tokens'] = df['review'].apply(lambda x: len(x.split()))


##################
import langid
import random

# Set the random seed for reproducibility
random.seed(42)

# Create a new 'language' column using langid
df['language'] = df['review'].apply(lambda x: langid.classify(x)[0])

# Remove any leading or trailing white spaces from the language labels
df['language'] = df['language'].str.strip()







######################

from sklearn.feature_extraction.text import CountVectorizer

# Define batch size for vectorization
batch_size = 1000

# Define a custom Dataset class
class AmazonReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, reviews):
        self.reviews = reviews

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        return self.reviews[idx]

# Create the dataset and data loader for vectorization
dataset = AmazonReviewsDataset(df['review'])
dataloader = DataLoader(dataset, batch_size=batch_size)

# Create the CountVectorizer
vectorizer = CountVectorizer(max_features=1000)

# Initialize an empty list to store the vectorized representations
vectorized_reviews = []

# Iterate over the batches of reviews
for batch in dataloader:
    # Perform vectorization
    X_batch = vectorizer.transform(batch).toarray()
    vectorized_reviews.append(X_batch)

# Concatenate the vectorized batches
X = np.concatenate(vectorized_reviews, axis=0)

# Create a new DataFrame with the transformed vectors
df_transformed = pd.DataFrame(X, columns=vectorizer.get_feature_names())

# Concatenate the transformed DataFrame with the original DataFrame
df = pd.concat([df, df_transformed], axis=1)

# Transform the 'language' feature to binary, where 1 indicates English and 0 indicates any other language
df['language'] = df['language'].apply(lambda x: 1 if x == 'en' else 0)

# Define the feature matrix X and the target variable y
X = df.drop(['score', 'review'], axis=1)
y = df['score']



##################


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)









