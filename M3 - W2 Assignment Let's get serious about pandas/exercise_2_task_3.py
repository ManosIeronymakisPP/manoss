import pandas as pd

# Load the dataset
data = {'name': ['Alice', 'Bob', 'Charlie', 'David','Bob','Charlie','David','Bob','Charlie','David'], 
        'course': ['math', 'math', 'math','math','science','science','science','history','history','history'], 
        'grade': [90, 85, 92, 88,70,80,82,95,91,90]}
df = pd.DataFrame(data)
"""
Create a MultiIndex
"""
df = df.set_index(['name', 'course'])
"""
Group the data by both the student's name and course, and calculate the average grade for each group
"""
grouped_df = df.groupby(level=['name', 'course']).mean()

"""
Select the rows corresponding to the students who have grade above 90 in course 'math'
"""
result = grouped_df[grouped_df.index.get_level_values('course') == 'math'].loc[grouped_df['grade'] > 90]

