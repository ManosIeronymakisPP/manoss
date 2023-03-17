import pandas as pd

data = {'name': ['Alice', 'Bob', 'Charlie', 'David','Bob','Charlie','David','Bob','Charlie','David'], 
        'course': ['math', 'math', 'math','math','science','science','science','history','history','history'], 
        'grade': [90, 85, 92, 88,70,80,82,95,91,90]}
df = pd.DataFrame(data)
"""
Create a MultiIndex using the 'name' and 'course' columns
"""
df = df.set_index(['name', 'course'])

"""
Reset the MultiIndex and set the 'name' column as the index
"""
df = df.reset_index().set_index('name')
