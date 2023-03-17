from typing import List, Tuple
import pandas as pd
import os
def avg_age_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    We create a pivot table that takes sex,pclass and embarked columns as index, age as value 
    and it returns a table in which we can clearly see the average age of each sex in which pclass they belonged and in which port they embarked.

    """
    pivot_table = df.pivot_table(index=["sex","pclass","embarked"], 
                                  values="age", 
                                  aggfunc="mean")
    
    return pivot_table









#we run our excel file with pandas
os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\module3")
filepath =  "titanic3.xls"
df = pd.read_excel(filepath)
#we call our function
print(avg_age_by_group(df))