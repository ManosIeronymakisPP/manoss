#First Dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

def print_data_info(file_path: str) -> None:
    """
    Prints information about the number of rows and columns in the dataset located at the given file path.
    
    Args:
    file_path: A string representing the file path of the dataset.
    
    Returns:
    None
    """
    # Importing the first dataset from the specified file path
    data = pd.read_excel(file_path)

    # Display the number of rows and columns in the dataset
    print(f'The dataset has {data.shape[0]} rows and {data.shape[1]} columns.')
    

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the given dataframe by performing the following operations:
    1. Removes duplicates
    2. Replaces all missing values with NaN
    3. Drops rows with missing values
    4. Removes scrapped data by dropping rows with invalid values
    5. Fixes encoding issues
    
    Args:
    data: A pandas dataframe to be cleaned.
    
    Returns:
    A cleaned pandas dataframe.
    """
    # Remove duplicates
    data.drop_duplicates(inplace=True)

    # Replace all missing values with NaN
    data.fillna(value=np.nan, inplace=True)

    # Drop rows with missing values
    data.dropna(inplace=True)

    # Remove scrapped data by dropping rows with invalid values
    data = data[data['issn'].str.contains(r'\d{4}-\d{3}[\dxX]$', na=False)]
    data = data[data['url'].str.startswith('http', na=False)]

    # Fix encoding issues
    data['journal_name'] = data['journal_name'].str.encode('utf-8').str.decode('ascii', 'ignore')

    return data


# Load the data from the specified file path and store it in a pandas dataframe
data: pd.DataFrame = pd.read_excel(r'C:\Users\Takhs\Downloads\module_project.xlsx')

# Clean the data
cleaned_data: pd.DataFrame = clean_data(data)

###NOW SOME EDA###

# Print the first 5 rows of the dataset
print(cleaned_data.head())

# Print the number of rows and columns in the dataset
print('Number of rows:', cleaned_data.shape[0])
print('Number of columns:', cleaned_data.shape[1])

# Display some basic descriptive statistics for numerical columns in the cleaned dataframe
print(cleaned_data.describe())

# Display the data types of each column in the cleaned dataframe
print(cleaned_data.dtypes)

##################### Second Dataset ###########################################


def print_data_info(file_path: str) -> None:
    """
    Prints information about the number of rows and columns in a dataset.

    Parameters:
        file_path (str): The file path of the dataset.

    Returns:
        None
    """
    # Importing the dataset from the specified file path
    data = pd.read_excel(file_path)

    # Showing information about the number of rows and columns
    print(f'The dataset has {data.shape[0]} rows and {data.shape[1]} columns.')


print_data_info(r'C:\Users\Takhs\Downloads\project_module1.xlsx')

# 1) Load the data into a pandas dataframe
data = pd.read_excel(r'C:\Users\Takhs\Downloads\project_module1.xlsx')

# 2) Remove duplicates
data.drop_duplicates(inplace=True)

# Replace all missing values with NaN
data.fillna(value=np.nan, inplace=True)

# 3) Drop rows with missing values
data.dropna(inplace=True)

# 4) Remove scrapped data by dropping rows with invalid values
data = data[data['issn'].str.contains(r'\d{4}-\d{3}[\dxX]$', na=False)]


# 5) Fix encoding issues
data['journal_name'] = data['journal_name'].str.encode('utf-8').str.decode('ascii', 'ignore')

# Print the number of rows and columns in the dataset
print('Number of rows:', data.shape[0])
print('Number of columns:', data.shape[1])

# Print some basic descriptive statistics for numerical columns
print(data.describe())

# Print the data types of each column
print(data.info)

# Print the number of unique values for each column
print(data.nunique())

# Select the columns of interest
columns_of_interest = ['citation_count_sum', 'paper_count_sum']
subset_data = data[columns_of_interest]

# Histogram of citation count sum
sns.histplot(data=subset_data, x="citation_count_sum")
plt.title("Histogram of Citation Count Sum")
plt.show()

# Scatterplot of citation count sum vs. paper count sum
sns.scatterplot(data=subset_data, x="citation_count_sum", y="paper_count_sum")
plt.title("Scatterplot of Citation Count Sum vs. Paper Count Sum")
plt.show()

# Boxplot of paper count sum
sns.boxplot(data=subset_data, y="paper_count_sum")
plt.title("Boxplot of Paper Count Sum")
plt.show()

########################### Third Dataset ############################################

def print_data_info(file_path: str) -> None:
    """
    Prints information about the number of rows and columns in a dataset.

    Parameters:
        file_path (str): The file path of the dataset.

    Returns:
        None
    """
    # Load the dataset
    data = pd.read_excel(file_path)

    # Showing information about the number of rows and columns
    print(f'The dataset has {data.shape[0]} rows and {data.shape[1]} columns.')


print_data_info(r'C:\Users\Takhs\Downloads\project_module2.xlsx')

# 1) Load the data into a pandas dataframe
data: pd.DataFrame = pd.read_excel(r'C:\Users\Takhs\Downloads\project_module2.xlsx')

# 2) Remove duplicates
data.drop_duplicates(inplace=True)

# Replace all missing values with NaN
data.fillna(value=np.nan, inplace=True)

# 3) Drop rows with missing values
data.dropna(inplace=True)

# Print the first 5 rows of the dataset
print(data.head())

# Print the number of rows and columns in the dataset
print('Number of rows:', data.shape[0])
print('Number of columns:', data.shape[1])

# Print some basic descriptive statistics for numerical columns
print(data.describe())

# Print the data types of each column
print(data.dtypes)

# Print the number of unique values for each column
print(data.nunique())