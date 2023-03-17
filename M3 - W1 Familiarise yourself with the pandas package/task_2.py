import os
import pandas as pd

def find_highest_expression(df):
    """
    first we calculate the average expression of the genes tp53 and foxo1 by taking the mean of their values along axis 1.
    The resulting mean values are then added to the DataFrame as a new column named "average_expression".
    Next we use ''max'' method to find the maximum value of ''average_expresion''.
    With the highest expression value, we can then extract all the rows from the DataFrame that have this highest value, by using boolean indexing. 
    The resulting DataFrame will only have one row, as the highest expression value should only occur once.
    Finally, we extract the cohort and tumor stage values from this single-row DataFrame and return them as a tuple.
    """
    df['average_expression'] = df[['tp53', 'foxo1']].mean(axis=1)
    max_expression = df['average_expression'].max()
    highest_expression_df = df[df['average_expression'] == max_expression]
    cohort = highest_expression_df['cohort'].iloc[0]
    tumor_stage = highest_expression_df['tumor_stage'].iloc[0]
    return cohort, tumor_stage








# prject directory
current_folder = os.getcwd()
desktop_path = "C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\"
# desktop_fldr = os.path.join(user_fldr, "OneDrive - Probability\Bureaublad")

# define the filename
filename = "METABRIC_RNA_Mutation.csv"

# Read data using pandas
df = pd.read_csv(os.path.join(desktop_path, filename))
print(find_highest_expression(df))