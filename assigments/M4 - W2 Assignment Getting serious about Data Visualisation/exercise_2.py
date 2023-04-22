import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

# Change the current working directory to the desired path
os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad")

# Set the file path to read the data from
filepath: str = "IMDB top 1000.csv"

# Read the data from the CSV file into a pandas DataFrame
df: pd.DataFrame = pd.read_csv(filepath)

# Remove duplicate rows based on the 'Title' column
df_clean: pd.DataFrame = df.drop_duplicates(subset='Title', keep='first')

# Split the 'Info' column into 'Votes' and 'Gross' columns
df_clean[['Votes', 'Gross']] = df_clean['Info'].str.split('|', expand=True)
df_clean['Votes'] = df_clean['Votes'].str.replace('Votes:', '').str.replace(',', '').astype(int)
df_clean['Gross'] = df_clean['Gross'].str.replace('Gross:', '').str.replace('$', '').str.replace('M', '').astype(float)

# Drop the original 'Info' column
df_clean.drop('Info', axis=1, inplace=True)

# Save the cleaned DataFrame to a new file
df_clean.to_csv('imdb_top_1000_cleaned.csv', index=False)

# Change the format of Votes column to include thousands separator
df_clean['Votes'] = df_clean['Votes'].apply(lambda x: '{:,}'.format(x).replace(',', '.'))

# Remove 'min' indication from Duration column and convert to integer data type
df_clean['Duration'] = df_clean['Duration'].str.replace(' min', '').astype(int)

# Drop all rows that contain NaN values
df_clean = df_clean.dropna()

# Rename the first column to "index"
df_clean = df_clean.rename(columns={'Unnamed: 0': 'index'})

# Reset the index to start from 0
df_clean = df_clean.reset_index(drop=True)

# Remove non-numeric characters from the Votes column
df_clean['Votes'] = df_clean['Votes'].str.replace('.', '').str.replace(',', '')

# Convert the Votes column to integer data type
df_clean['Votes'] = df_clean['Votes'].astype(int)

# Create a pairplot to visualize the pairwise relationships between the variables, colored by certificate
sns.pairplot(df_clean[['Rate', 'Metascore', 'Votes', 'Gross', 'Certificate']], hue='Certificate')

# Create a scatter plot of Votes vs Gross, colored by certificate
sns.lmplot(x='Votes', y='Gross', hue='Certificate', data=df_clean)

# Create a heatmap to visualize the correlation matrix
sns.heatmap(df_clean[['Rate', 'Metascore', 'Votes', 'Gross']].corr(), annot=True, cmap='coolwarm')

# Display the plots
plt.show()

# Create a heatmap to visualize the correlation matrix between Rate and Duration column
sns.heatmap(df_clean[['Rate',"Duration"]].corr(), annot=True, cmap='coolwarm')

# Display the plots
plt.show()

# Convert the Duration column to integer data type
df_clean['Duration'] = df_clean['Duration'].astype(int)

# Create a scatterplot to visualize the relationship between Duration and Rate
sns.scatterplot(x='Duration', y='Rate', data=df_clean, hue='Certificate')

# Set the x-axis range to display only the first, middle, and last values of Duration
dur_min: int = df_clean['Duration'].min()
dur_max: int = df_clean['Duration'].max()
dur_range: List[int] = [dur_min, (dur_min+dur_max)//2, dur_max]
plt.xticks(dur_range)


# Show the legend
plt.show()


##################################### COMMENT AREA #####################################################################################################################################   

#After conducting data cleaning, a visual analysis was performed on the variables in the dataset, including Certificate, Duration, Rate, Metascore, Votes, and Gross. 
#Upon examination of the columns Title, Genre, Description, and Cast, there appeared to be no discernible correlation between these variables and the Rate column.
#A series of plots and a heatmap were then generated to determine which factors were most strongly correlated with a movie's IMDB rating. 
#The analysis revealed that the three most highly correlated variables with the Rate column were Votes, Duration, and Gross. 
#The Votes variable demonstrated the strongest correlation with a coefficient of 0.64, followed by Duration with a coefficient of 0.27, and lastly Gross with a coefficient of 0.17.
#The Votes column represents the number of individuals who voted for a particular movie, while the Duration column denotes the length of time in minutes that a given movie runs.
#The Gross column indicates the gross earnings for a particular movie in U.S. dollars.






























