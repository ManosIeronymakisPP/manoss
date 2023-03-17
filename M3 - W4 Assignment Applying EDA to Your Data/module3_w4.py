import pandas as pd
import os

#open the xls file
os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad")
filepath =  "Sample - Superstore.xls"
df = pd.read_excel(filepath)

#show information about the number of rows and columns (shape of the data
num_rows, num_cols = df.shape
print(f'The DataFrame has {num_rows} rows and {num_cols} columns.')

# Compute the mean and median values of the numerical columns
mean_values = df[['Row ID', 'Postal Code', 'Sales', 'Quantity', 'Discount', 'Profit']].mean()
median_values = df[['Row ID', 'Postal Code', 'Sales', 'Quantity', 'Discount', 'Profit']].median()

print('Mean values:')
print(mean_values)

print('\nMedian values:')
print(median_values)

# Compute the unique values of the categorical columns
unique_values = df[['Ship Mode', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Region', 'Category', 'Sub-Category', 'Product Name']].apply(pd.Series.nunique)

print('\nUnique values:')
print(unique_values)


# Plot the numerical columns
df[['Row ID', 'Postal Code', 'Sales', 'Quantity', 'Discount', 'Profit']].plot(kind='density', subplots=True, layout=(2,3), sharex=False)

#create and display the correlation matrix
correlation_matrix = df[['Row ID', 'Postal Code', 'Sales', 'Quantity', 'Discount', 'Profit']].corr()
print(correlation_matrix)
