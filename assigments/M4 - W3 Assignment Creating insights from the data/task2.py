#import libraries
import pandas as pd
import os
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad")
filepath =  "data.xlsx"
df = pd.read_excel(filepath)



# check if any column has NaN values
if df.isna().any().any():
    print("There are NaN values in the DataFrame.")
else:
    print("There are no NaN values in the DataFrame.")
    
    
    
# print the index values of rows with NaN values
nan_rows = df[df.isna().any(axis=1)].index.tolist()
print("Rows with NaN values:", nan_rows)

# drop rows with NaN values
df = df.dropna()

# filter rows to keep only those where country is 'GR'
df = df[df['country'] == 'GR']

# filter rows to remove those where gender is 0 or 3
df = df[~df['gender'].isin([0, 3])]

# select only the columns A3, B3, C2, D3, E3 and gender
df = df.loc[:,["A3","B3","C2","D3","E3","gender"]]


# filter out participants who are not Male (gender = 1) or Female (gender = 2)
df = df[df['gender'].isin([1, 2])]

# separate the data for men and women into separate DataFrames
men_df = df[df['gender'] == 1]
women_df = df[df['gender'] == 2]

# perform a t-test for each of the five questions
for question in ['A3', 'B3', 'C2', 'D3', 'E3']:
    men_responses = men_df[question]
    women_responses = women_df[question]
    t_statistic, p_value = ttest_ind(men_responses, women_responses)
    print('Question {}: t-statistic={}, p-value={}'.format(question, t_statistic, p_value))
    
    
    
 
# melt the dataframe to long format
df_qm = df.melt(id_vars=['gender'], var_name='question', value_name='response')


# create separate dataframes for men and women
men_df = df[df['gender'] == 1]
women_df = df[df['gender'] == 2]

# create a barplot with both men and women's data
sns.catplot(x='uestion', y='response', hue='gender', data=df_qm, kind='bar', alpha=0.8, ci=None)
plt.title('Responses by Question and Gender')
plt.show()

############################################# RESULTS ############################################################################################################################################

#For Question A3, the t-statistic is negative (-1.85) and the p-value is 0.066, which suggests that there is some evidence of a difference between men and women's responses, but it is not statistically significant at the conventional level of 0.05.
#For Question B3, the t-statistic is positive (0.15) and the p-value is 0.88, which suggests that there is not a statistically significant difference between men and women's responses.
#For Question C2, the t-statistic is positive (1.28) and the p-value is 0.20, which suggests that there is not a statistically significant difference between men and women's responses.
#For Question D3, the t-statistic is positive (1.15) and the p-value is 0.25, which suggests that there is not a statistically significant difference between men and women's responses.
#For Question E3, the t-statistic is negative (-0.67) and the p-value is 0.50, which suggests that there is not a statistically significant difference between men and women's responses.
#Overall, based on these results, we can conclude that there is no strong evidence of a significant difference in responses between men and women for any of the questions. However, for Question A3, there is some evidence of a difference that falls just short of statistical significance.

    
    
    
    