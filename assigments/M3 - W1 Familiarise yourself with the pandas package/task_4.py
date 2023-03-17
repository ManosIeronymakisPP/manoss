import pandas as pd
import os

def missing_data_columns(file_path):
    """"
    we returns a boolean series indicating which columns contain missing data. 
    This series is then passed to the list function to return a list of column names with missing data.
    """
    return "Columns with missing data:",[ list(df.columns[df.isnull().any()])]


# prject directory
current_folder = os.getcwd()
desktop_path = "C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\"
# desktop_fldr = os.path.join(user_fldr, "OneDrive - Probability\Bureaublad")

# define the filename
filename = "METABRIC_RNA_Mutation.csv"

# Read data using pandas
df = pd.read_csv(os.path.join(desktop_path, filename))
print(missing_data_columns(df))
