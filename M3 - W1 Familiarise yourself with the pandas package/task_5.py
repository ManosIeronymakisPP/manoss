import os
import pandas as pd

def find_missing_data(df: pd.DataFrame) -> None:
    """"
    we filter the rows in the Df that have missing values in either the 'tumor_size' or 'mutation_count' columns using the isnull method and the | operator in order to find the missing data.
    after we extract the values in the 'patient_id' column for the rows in missing_data using the values attribute and store the result in a list called missing_patient_ids.
    after we extract the values in the 'cohort' column for the rows in missing_data using the values attribute and store the result in a list called missing_cohorts.
    """
    
    
    missing_data = df[(df['tumor_size'].isnull()) | (df['mutation_count'].isnull())]
    missing_patient_ids = list(missing_data['patient_id'].values)
    missing_cohorts = list(missing_data['cohort'].values)
    print(f"Patient IDs with missing data: {missing_patient_ids}")
    print(f"Corresponding cohorts: {missing_cohorts}")




# prject directory
current_folder = os.getcwd()
desktop_path = "C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\"
# desktop_fldr = os.path.join(user_fldr, "OneDrive - Probability\Bureaublad")

# define the filename
filename = "METABRIC_RNA_Mutation.csv"

# Read data using pandas
df = pd.read_csv(os.path.join(desktop_path, filename))
print(find_missing_data(df))