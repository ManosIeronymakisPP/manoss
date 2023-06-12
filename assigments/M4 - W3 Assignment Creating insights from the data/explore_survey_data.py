#import libraries
import pandas as pd
import os
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from Excel file and return as a pandas DataFrame.
    
    Args:
        filepath (str): Path to the Excel file.
    
    Returns:
        pandas DataFrame: Loaded data.
    """
    df = pd.read_excel(filepath)
    return df

def check_nan_values(df: pd.DataFrame) -> None:
    """
    Check if any column has NaN values and print the result.
    Also print the index values of rows with NaN values, if any.
    
    Args:
        df (pandas DataFrame): Input data.
    
    Returns:
        None
    """
    if df.isna().any().any():
        print("There are NaN values in the DataFrame.")
        nan_rows = df[df.isna().any(axis=1)].index.tolist()
        print("Rows with NaN values:", nan_rows)
    else:
        print("There are no NaN values in the DataFrame.")

def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter data to keep only the relevant rows.
    
    Args:
        df (pandas DataFrame): Input data.
    
    Returns:
        pandas DataFrame: Filtered data.
    """
    df = df.dropna() # drop rows with NaN values
    df = df[df['country'] == 'GR'] # keep only rows with 'GR' in the 'country' column
    df = df[~df['gender'].isin([0, 3])] # remove rows with gender 0 or 3
    df = df.loc[:,["A3","B3","C2","D3","E3","gender"]] # select only the relevant columns
    df = df[df['gender'].isin([1, 2])] # keep only rows with gender 1 or 2
    return df

def perform_t_test(df: pd.DataFrame) -> None:
    """
    Perform t-test for each question and print the result.
    
    Args:
        df (pandas DataFrame): Input data.
    
    Returns:
        None
    """
    for question in ['A3', 'B3', 'C2', 'D3', 'E3']:
        men_responses = df[df['gender'] == 1][question]
        women_responses = df[df['gender'] == 2][question]
        t_statistic, p_value = ttest_ind(men_responses, women_responses)
        print(f'Question {question}: t-statistic={t_statistic:.3f}, p-value={p_value:.3f}')

def plot_responses(df: pd.DataFrame) -> None:
    """
    Create a barplot with both men and women's data and show the plot.
    
    Args:
        df (pandas DataFrame): Input data.
    
    Returns:
        None
    """
    df_qm = df.melt(id_vars=['gender'], var_name='question', value_name='response') # melt the DataFrame to long format
    sns.catplot(x='question', y='response', hue='gender', data=df_qm, kind='bar', alpha=0.8, ci=None)
    plt.title('Responses by Question and Gender')
    plt.show()

if __name__ == "__main__":
    filepath = "C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\data.xlsx"
    df = load_data(filepath)
    check_nan_values(df)
    df = filter_data(df)
    perform_t_test(df)
    plot_responses(df)


############################################# RESULTS ############################################################################################################################################
#For Question A3, the t-statistic is negative (-1.85) and the p-value is 0.066, which suggests that there is some evidence of a difference between men and women's responses, but it is not statistically significant at the conventional level of 0.05.
#For Question B3, the t-statistic is positive (0.15) and the p-value is 0.88, which suggests that there is not a statistically significant difference between men and women's responses.
#For Question C2, the t-statistic is positive (1.28) and the p-value is 0.20, which suggests that there is not a statistically significant difference between men and women's responses.
#For Question D3, the t-statistic is positive (1.15) and the p-value is 0.25, which suggests that there is not a statistically significant difference between men and women's responses.
#For Question E3, the t-statistic is negative (-0.67) and the p-value is 0.50, which suggests that there is not a statistically significant difference between men and women's responses.
#Overall, based on these results, we can conclude that there is no strong evidence of a significant difference in responses between men and women for any of the questions. However, for Question A3, there is some evidence of a difference that falls just short of statistical significance.

