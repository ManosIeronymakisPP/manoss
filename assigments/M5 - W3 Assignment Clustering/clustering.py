# Import necessary libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Change the current working directory
os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad\\ELU\\M5 - W3 Assignment Clustering")

# Define the file path
filepath =  "Mall_Customers.csv"


# Load the dataset into a pandas DataFrame
df = pd.read_csv(filepath)

# Check for any missing or null data points
print("Missing values in each column:\n", df.isnull().sum())

# Plot the distribution of gender
sns.countplot(x='Gender', data=df)
plt.title('Distribution of Gender')
plt.show()

# Create histograms for all numeric columns
for column in df.select_dtypes(include=[np.number]).columns:
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

sns.pairplot(df[["Age","Annual Income (k$)","Spending Score (1-100)"]])






# Define the columns to be scaled
cols_to_scale = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Initialize the Scaler
scaler = StandardScaler()

# Scale the selected columns and replace in the dataframe
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])


#Create and fit the K-Means model for a range of cluster sizes
wcss = []  # Within-Cluster-Sum-of-Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)  # Inertia: Sum of distances of samples to their closest cluster center

#Plot the WCSS for each number of clusters to visualize the 'Elbow'
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('K-Means Clustering: The Elbow Method')
plt.show()

#Create and fit the final K-Means model using the optimal number of clusters

# Suppose the optimal number of clusters is 3
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(df)

#Assign the samples to the computed clusters
df['Cluster'] = kmeans.labels_





# Import necessary libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Plot the dendrogram to find optimal number of clusters
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(sch.linkage(df, method='ward'))  # 'ward' minimizes variance of the clusters being merged
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# Suppose we decided that the optimal number of clusters is 3, from observing the dendrogram
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

# Fit the model and assign each sample to a cluster
df['Cluster'] = hc.fit_predict(df)

# Now, you can analyze the resulting clusters








