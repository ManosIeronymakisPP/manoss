import pandas as pd
import os




def create_age_range_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    we create a new column in our file called "Age_Range" and at the same time we categorize the ages into bins. 
    After we print the unique values of age range and we match and print the average fare to our age range.
    """
    df["Age_Range"] = pd.cut(df["age"], bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80], 
                            labels=["0-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-55", "56-60", "61-65", "66-70", "71-75", "76-80"])
    print( "the unique values af Age_Range are:",  df["Age_Range"].unique())     
    print(df.groupby("Age_Range")["fare"].mean().reset_index()) 
    



#we read our excel file with python pandas
os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\module3")
filepath =  "titanic3.xls"
df = pd.read_excel(filepath)
# we call our function
print(create_age_range_column(df))