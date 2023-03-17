import pandas as pd

df = pd.read_csv('METABRIC_RNA_Mutation.csv' ,sep=',',header=0,low_memory=False)

print(df.to_string()) 
