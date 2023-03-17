from typing import List, Tuple
import pandas as pd
import os
def clean_and_analyze(df: pd.DataFrame) -> pd.DataFrame:
    """
    We fill any missing values in the 'Embarked' column.
    """
    value_counts = df['embarked'].value_counts()
    most_common_value = value_counts.index[0]
    df['embarked'].fillna(most_common_value, inplace=True)
    
    """
    We drop the 'Cabin' and 'Ticket' columns.
    """
    df.drop(["cabin", "ticket"],inplace=True, axis=1)
    """
    We group the data by the 'Embarked' column.
    """
    grouped = df.groupby('embarked')
    """
    We calculate the number of passengers for each port
    """
    count = grouped['survived'].count()
    """
    We calculate the survival rate for each port
    """
    survival_rate = grouped['survived'].mean()
    """
    We combine the count and survival rate into a single DataFrame
    """
    result = pd.DataFrame({'Passenger Count': count, 'Survival Rate': survival_rate})
    return result

#we run our excel file with pandas
os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\module3")
filepath =  "titanic3.xls"
df = pd.read_excel(filepath)
#we call our function
print(clean_and_analyze(df))

