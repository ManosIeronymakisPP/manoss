import pandas as pd

# Load the dataset
data = {'name': ['Alice', 'Bob', 'Charlie', 'David','Bob','Charlie','David','Bob','Charlie','David'], 
        'course': ['math', 'math', 'math','math','science','science','science','history','history','history'], 
        'grade': [90, 85, 92, 88,70,80,82,95,91,90]}
"""
cretaing a multindex with name and course column
"""
df = pd.DataFrame(data)
df = df[['name', 'course']]
