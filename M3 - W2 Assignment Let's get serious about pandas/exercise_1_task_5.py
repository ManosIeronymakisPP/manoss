from typing import List, Tuple
import pandas as pd
import os


def analyze_fares(df: pd.DataFrame) -> pd.DataFrame:
    """
     Bin the 'Fare' column into 4 equal bins and create a new column 'FareRange'
    """
    df['FareRange'] = pd.cut(df['fare'], bins=4, labels=["Low", "Medium", "High", "Very High"])
    """
    Group the dataframe by the 'FareRange' and 'Sex' columns
    """
    grouped = df.groupby(['FareRange', 'sex'])
    """
    Calculate the mean, min, max, and standard deviation of the 'Age' column
    """
    result = grouped['age'].agg(['mean', 'min', 'max', 'std'])
    return result


#we open our excel file with pandas
os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\module3")
filepath =  "titanic3.xls"
df = pd.read_excel(filepath)
#we run our function
print(analyze_fares(df))

