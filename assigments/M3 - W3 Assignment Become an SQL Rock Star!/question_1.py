import pandas as pd
df = pd.read_csv('METABRIC_RNA_Mutation.csv' ,sep=',',header=0,low_memory=False)
df.groupby(['death_from_cancer', 'tumor_stage'])['tumor_size'].mean()

