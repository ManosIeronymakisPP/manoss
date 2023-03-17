import pandas as pd

# Load the dataset
data = {'name': ['Alice', 'Bob', 'Charlie', 'David','Bob','Charlie','David','Bob','Charlie','David'], 
        'course': ['math', 'math', 'math','math','science','science','science','history','history','history'], 
        'grade': [90, 85, 92, 88,70,80,82,95,91,90]}
df = pd.DataFrame(data)

"""
Group the data by both the student's name and course
"""
grouped_data = df.groupby(['name', 'course'])
"""
Calculate the average grade for each group
"""
result = grouped_data['grade'].mean()
