#import libraries
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import preprocessing

###QUESTION1###

# Set the working directory and load the dataset
os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\ELU\\M4 - W4 Assignment Use statistical thinking and methods for modelling")
filepath = "The Boston Housing Dataset.xlsx"
df = pd.read_excel(filepath)

# Rename the columns to make them more readable
df.columns= ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B - 1000(Bk - 0.63)^2","LSTAT","MEDV"]

# Check for any missing values in the dataset
df.isnull().any()

# Display the data type of each column in the dataset
df.dtypes

# Print the number of rows and columns in the dataset
np.shape(df)

# Display summary statistics of the dataset
df.describe()

# Plot the dataset using boxplots to detect outliers
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
for i, ax in enumerate(axs.flatten()):
    if i < len(df.columns):
        sns.boxplot(y=df.columns[i], data=df, ax=ax)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

# Display the plot
plt.show()

# Calculate the percentage of outliers in each numerical column of the dataset
def get_outlier_percentage(col, low_percentile=0.01, high_percentile=0.99):
    """
    This function calculates the percentage of outliers in a given column of a dataset.
    """
    low, high = df[col].quantile([low_percentile, high_percentile])
    num_outliers = len(df[(df[col] < low) | (df[col] > high)])
    return num_outliers / len(df) * 100

# Apply the function to each numerical column in the dataset and print the results with comments
for col in df.columns:
    # Only consider numerical columns
    if df[col].dtype != object:
        # Get the outlier percentage for the column
        outlier_percentage = get_outlier_percentage(col)
        # Print the column name and the outlier percentage
        print(f"{col}: {outlier_percentage:.2f}% outliers")

# Plot the distribution of all the columns in the dataset
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in df.items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

# Plot the correlation matrix of the dataset
plt.figure(figsize=(20, 10))
sns.heatmap(df.corr().abs(),  annot=True)

# Identify the columns with high correlation with MEDV
# LSTAT, INDUS, RM, TAX, NOX, PTRATIO have good correlation with MEDV
# Plot these columns against MEDV


# Scale the columns before plotting them against MEDV
min_max_scaler = preprocessing.MinMaxScaler()
column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
x = df.loc[:,column_sels]
y = df['MEDV']
x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for i, k in enumerate(column_sels):
    sns.regplot(y=y, x=x[k], ax=axs[i])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

###QUESTION2###
# Set the working directory and load the dataset
os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\ELU\\M4 - W4 Assignment Use statistical thinking and methods for modelling")
filepath = "The Boston Housing Dataset.xlsx"
df = pd.read_excel(filepath)

# Rename the columns to make them more readable
df.columns= ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B - 1000(Bk - 0.63)^2","LSTAT","MEDV"]


# Select features (columns) and target (MEDV)
X = df[['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']]
y = df['MEDV']

# Split data into training and testing sets with a 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

###QUESTION3###
# Set the working directory and load the dataset
os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\ELU\\M4 - W4 Assignment Use statistical thinking and methods for modelling")
filepath = "The Boston Housing Dataset.xlsx"
df = pd.read_excel(filepath)

# Rename the columns to make them more readable
df.columns= ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B - 1000(Bk - 0.63)^2","LSTAT","MEDV"]



# define X and y
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scale the training and testing data
min_max_scaler = preprocessing.MinMaxScaler()
X_train = pd.DataFrame(data=min_max_scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(data=min_max_scaler.transform(X_test), columns=X_test.columns)

# fit the model on the training data
lm = LinearRegression()
lm.fit(X_train, y_train)

# print the model coefficients
print(lm.coef_)

# make predictions on the testing data
y_pred = lm.predict(X_test)

# calculate and print adjusted R-squared and mean squared error
n = len(X_test)
p = len(X_test.columns)
r_squared = r2_score(y_test, y_pred)
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
mse = mean_squared_error(y_test, y_pred)

print(f"Adjusted R-squared: {adjusted_r_squared:.3f}")
print(f"Mean squared error: {mse:.3f}")

###QUESTION4###

# Based on the model results, the final model's adjusted R-squared is 0.682, 
# which suggests that the model explains 68.2% of the variance in the target variable (housing prices). 
# This value is fairly high, indicating that the model is a reasonably good fit for the data.

# The mean squared error (MSE) of the model is 18.867, which indicates the average squared difference between the predicted and actual values. 
# While this value depends on the scale of the target variable, a lower MSE is generally desirable, so the model's MSE could be further improved.

# Looking at the coefficients for each predictor, we can gain insights into their relationships with the target variable. 
# For example, the RM (number of rooms per dwelling) predictor has the highest coefficient of 19.51858372, 
# suggesting that more rooms generally lead to higher housing prices. 
# On the other hand, the LSTAT (percentage of lower status of the population) predictor has a coefficient of -20.73537433, 
# indicating that areas with a higher percentage of lower status population tend to have lower housing prices.

# Overall, the model seems to be a reasonable fit for the data, although there is room for improvement. 
# The coefficients provide insights into the relationships between the predictors and the target variable, 
# which can be useful for making predictions and understanding the factors that influence housing prices.


###QUESTION5###
# Set the working directory and load the dataset
os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\ELU\\M4 - W4 Assignment Use statistical thinking and methods for modelling")
filepath = "The Boston Housing Dataset.xlsx"
df = pd.read_excel(filepath)

# Rename the columns to make them more readable
df.columns= ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B - 1000(Bk - 0.63)^2","LSTAT","MEDV"]


# define X and y
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# This code is similar to the previous one, but after splitting the data into training and testing sets, 
# we scale the entire dataset using the MinMaxScaler and fit the model on the entire dataset.

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scale the training and testing data
min_max_scaler = MinMaxScaler()
X_train = pd.DataFrame(data=min_max_scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(data=min_max_scaler.transform(X_test), columns=X_test.columns)
X = pd.DataFrame(data=min_max_scaler.transform(X), columns=X.columns)

# fit the model on the training data
lm = LinearRegression()
lm.fit(X_train, y_train)

# print the model coefficients
coef = pd.DataFrame(data=lm.coef_, index=X_train.columns, columns=['Coefficient'])
print(coef)

# make predictions on the testing data
y_pred = lm.predict(X_test)

# calculate and print adjusted R-squared and mean squared error
n = len(X_test)
p = len(X_test.columns)
r_squared = r2_score(y_test, y_pred)
adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
mse = mean_squared_error(y_test, y_pred)

print(f"Adjusted R-squared: {adjusted_r_squared:.3f}")
print(f"Mean squared error: {mse:.3f}")


# Comparing the previous results with the new results obtained by using the whole dataset, 
# we can see that there is no significant difference in the coefficients of the variables. 
# The Adjusted R-squared value remains the same at 0.682, which indicates that the model explains around 68% of the variance in the target variable. 
# The mean squared error also remains similar at 18.867.

# However, using the whole dataset instead of just the test dataset for fitting the model can lead to overfitting, 
# which is a condition where the model fits the training data too well but fails to generalize to new data. 
# Therefore, it is important to evaluate the model on a separate test dataset to ensure that it is not overfitting.















