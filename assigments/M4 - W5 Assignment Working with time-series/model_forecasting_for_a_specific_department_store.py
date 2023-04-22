import pandas as pd
from sklearn.metrics import mean_squared_error
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

# Load the data
os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\ELU\\M4 - W5 Assignment Working with time-series")
filepath =  "Features data set.csv"
filepath1 =  "sales data-set.csv"
filepath2 =  "stores data-set.csv"

# Loading the data into pandas dataframe for EDA
df_stores = pd.read_csv(filepath2)
df_features = pd.read_csv(filepath)
df_features.Date = pd.to_datetime(df_features.Date)
df_sales = pd.read_csv(filepath1)
df_sales.Date = pd.to_datetime(df_sales.Date)

#merging the dataframes
df_features = df_features.merge(df_stores, on = 'Store')
df = df_features.merge(df_sales, on = ['Store','Date','IsHoliday'])


# Remove the NaN values with 0
df=df.fillna(0)


# Select the data for a specific department and store
dept = 10
store = 20
df_new = df[(df.Dept == dept) & (df.Store == store)][["Date","Weekly_Sales","Store","Dept"]]

# Convert the Date column to datetime format
df_new['Date'] = pd.to_datetime(df_new['Date'])

# Extract useful features from the Date column
df_new['Year'] = df_new['Date'].dt.year
df_new['Month'] = df_new['Date'].dt.month
df_new['Day'] = df_new['Date'].dt.day
df_new['DayOfWeek'] = df_new['Date'].dt.dayofweek

# Split the data into features (X) and target (y)
X = df_new[['Year', 'Month', 'Day', 'DayOfWeek']]
y = df_new['Weekly_Sales']

# Split the data into training and testing sets using time series split
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Train a random forest regression model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)

# Interpret the model's predictions and provide insights for store managers
# For example, you could use the feature importances to identify which features are most important for predicting sales, and use this information to optimize inventory management or marketing strategies.
importances = rf.feature_importances_
print("Feature importances:", importances)