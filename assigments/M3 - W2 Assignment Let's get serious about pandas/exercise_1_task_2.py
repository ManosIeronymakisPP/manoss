from typing import List, Tuple
import pandas as pd
import os

def create_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    We create a new column called "FamilySize" that is the sum of the "SibSp" and "Parch" columns.
    After we create another column called "FamilySizeCategory" which is the same with family size column but in categories.
    In the end we create a pivot table that shows the rate of passengers for each familysizecategory that survived.

    """
    df["FamilySize"] = df["sibsp"] + df["parch"]
    df["FamilySizeCategory"] = pd.cut(df["FamilySize"], 
                                       bins=[-1, 0, 3, 6, 10], 
                                       labels=["0", "1-3", "4-6", "7+"])
    
    pivot_table = df.pivot_table(index="FamilySizeCategory", 
                                  values="survived", 
                                  aggfunc="mean")
    return pivot_table



#we read our excel file with pandas
os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\module3")
filepath =  "titanic3.xls"
df = pd.read_excel(filepath)
#we call our function
print(create_family_size(df))