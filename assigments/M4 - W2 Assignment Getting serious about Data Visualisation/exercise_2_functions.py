import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, List



def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the given DataFrame by removing duplicates, splitting the 'Info' column into 'Votes' and 'Gross'
    columns, dropping the 'Info' column, changing the format of the 'Votes' column, removing the 'min' indication
    from the 'Duration' column, removing rows with missing values, renaming the first column to 'index',
    resetting the index, removing non-numeric characters from the 'Votes' column, and converting the 'Votes'
    column to integer data type.
    """
    # Remove duplicates based on the 'Title' column
    df_clean = df.drop_duplicates(subset='Title', keep='first')
    
    # Split the 'Info' column into 'Votes' and 'Gross' columns
    df_clean[['Votes', 'Gross']] = df_clean['Info'].str.split('|', expand=True)
    df_clean['Votes'] = df_clean['Votes'].str.replace('Votes:', '').str.replace(',', '').astype(int)
    df_clean['Gross'] = df_clean['Gross'].str.replace('Gross:', '').str.replace('$', '').str.replace('M', '').astype(float)
    
    # Drop the original 'Info' column
    df_clean.drop('Info', axis=1, inplace=True)
    
    # Change the format of Votes column
    df_clean['Votes'] = df_clean['Votes'].apply(lambda x: '{:,}'.format(x).replace(',', '.'))
    
    # Remove min indication and turn the column into int 
    df_clean['Duration'] = df_clean['Duration'].str.replace(' min', '').astype(int)
    
    # Remove all the rows that have nah
    df_clean = df_clean.dropna()
    
    # Rename the first column to "index"
    df_clean = df_clean.rename(columns={'Unnamed: 0': 'index'})
    
    # Reset the index to start from 0
    df_clean = df_clean.reset_index(drop=True)
    
    # Remove non-numeric characters from the my_column column
    df_clean['Votes'] = df_clean['Votes'].str.replace('.', '').str.replace(',', '')
    
    # Convert the my_column column to integer data type
    df_clean['Votes'] = df_clean['Votes'].astype(int)
    
    return df_clean


def create_pairplot(df: pd.DataFrame) -> None:
    """
    Creates a pairplot to visualize the pairwise relationships between the variables, colored by certificate.
    """
    sns.pairplot(df[['Rate', 'Metascore', 'Votes', 'Gross', 'Certificate']], hue='Certificate')
    plt.show()


def create_scatterplot(df: pd.DataFrame) -> None:
    """
    Creates a scatter plot of Votes vs Gross, colored by certificate.
    """
    sns.lmplot(x='Votes', y='Gross', hue='Certificate', data=df)
    plt.show()


def create_heatmap(df: pd.DataFrame) -> None:
    """
    Creates a heatmap to visualize the correlation matrix.
    """
    sns.heatmap(df[['Rate', 'Metascore', 'Votes', 'Gross']].corr(), annot=True, cmap='coolwarm')
    plt.show()


def create_duration_rate_scatterplot(df: pd.DataFrame) -> None:
    """
    Creates a scatter plot of Duration vs Rate, colored by certificate.
    """
    sns.lmplot(x='Duration', y='Rate', hue='Certificate', data=df)
    plt.show()
    
if __name__ == '__main__':
    # Define the file path
    os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad")
    filepath =  "IMDB top 1000.csv"
    
    # Read the file
    df = pd.read_csv(filepath)
    
    # Clean the data
    df_clean = clean_dataframe(df)
    
    # Create the visualizations
    create_pairplot(df_clean)
    create_scatterplot(df_clean)
    create_heatmap(df_clean)
    create_duration_rate_scatterplot(df_clean)
    
    
    

##################################### COMMENT AREA #####################################################################################################################################   

#After conducting data cleaning, a visual analysis was performed on the variables in the dataset, including Certificate, Duration, Rate, Metascore, Votes, and Gross. 
#Upon examination of the columns Title, Genre, Description, and Cast, there appeared to be no discernible correlation between these variables and the Rate column.
#A series of plots and a heatmap were then generated to determine which factors were most strongly correlated with a movie's IMDB rating. 
#The analysis revealed that the three most highly correlated variables with the Rate column were Votes, Duration, and Gross. 
#The Votes variable demonstrated the strongest correlation with a coefficient of 0.64, followed by Duration with a coefficient of 0.27, and lastly Gross with a coefficient of 0.17.
#The Votes column represents the number of individuals who voted for a particular movie, while the Duration column denotes the length of time in minutes that a given movie runs.
#The Gross column indicates the gross earnings for a particular movie in U.S. dollars.


















