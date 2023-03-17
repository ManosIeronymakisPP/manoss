
import os
import pandas as pd



def calc_mean_tumour_size(df):
    """
    Parameters:
    df: pandas.DataFrame
        A dataframe containing columns 'vital_status' and 'tumor_stage', and 'tumor_size'

    Returns:
        A dataframe with indexes (vital_status, tumor_stage) and one column 'tumor_size_mean'
    """
    result = df.groupby(['death_from_cancer', 'tumor_stage'])['tumor_size'].mean().reset_index()
    result = result.set_index(['death_from_cancer', 'tumor_stage'])
    result.rename(columns={'tumor_size': 'tumor_size_mean'}, inplace=True)
    return result







# prject directory
current_folder = os.getcwd()
desktop_path = "C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\"
# desktop_fldr = os.path.join(user_fldr, "OneDrive - Probability\Bureaublad")

# define the filename
filename = "METABRIC_RNA_Mutation.csv"

# Read data using pandas
df = pd.read_csv(os.path.join(desktop_path, filename))
print(calc_mean_tumour_size(df))
