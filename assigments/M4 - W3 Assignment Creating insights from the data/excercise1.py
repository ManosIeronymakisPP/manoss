import pandas as pd
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

#import goodreads data, skipping a few corrupted rows (extra commas)
os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad")
filepath =  "books.csv"
df = pd.read_csv(filepath, error_bad_lines=False)
df.rename(columns={'  num_pages': 'num_pages'}, inplace=True)
##df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
df.drop_duplicates(inplace=True)


#find the most fammous language
plt.figure(figsize=(16,10))
ax = df.groupby('language_code')['title'].count().plot.bar()
plt.title('Code of the language')
plt.xticks(fontsize = 14)
for p in ax.patches:
    height = p.get_height()
    x_val = p.get_x()
    ax.annotate(str(height),(x_val-0.15, height+150))
    
    
#Rating Distribution
sns.distplot(df['average_rating'], bins=30)

#Distribution plot for number of pages(books).
sns.distplot(df['num_pages'],bins = 30)

# Top 20 Publisher
publisher = df.value_counts('publisher').sort_values(ascending=False).head(15)
sns.barplot(y=publisher.index,x = publisher)


#Most Occurring Books
plt.figure(figsize = (8,5))
title = df.value_counts('title').sort_values(ascending=False).head(20)
sns.barplot(y=title.index,x=title)
plt.title('Most Occurring Books')
plt.xlabel("Number of occurances")
plt.ylabel('Books')

#Top 10 highest page count books
x = df.loc[:,['title', 'num_pages']].sort_values(by=['num_pages'],ascending = False).head(10)
sns.barplot(y = 'title',x='num_pages',data = x)


#Highest 10 text review count books
x = df.loc[:,['title', 'text_reviews_count']].sort_values(by=['text_reviews_count'],ascending = False).head(10)
sns.barplot(y = 'title',x='text_reviews_count',data = x)


sns.scatterplot(y='text_reviews_count',x='average_rating',data=df)

sns.scatterplot(y='num_pages',x='average_rating',data=df)

sns.scatterplot(y='ratings_count',x='average_rating',data=df)

sns.scatterplot(x='ratings_count',y='text_reviews_count',data=df)




#In which year, highest books published.?
months=[]
days=[]
year=[]
for i in df['publication_date']:
    year.append(i.split('/')[2])
    months.append(i.split('/')[0])

df['year'] = year
df['month'] = months

plt.figure(figsize = (10,5))
year = df['year'].value_counts().head(10)
year
sns.barplot(y=year.index,x=year)


#In which month, highest number of books published.?
plt.figure(figsize = (10,5))
month = df['month'].value_counts().head(10)
sns.barplot(y=month.index,x=month)








# #Above Graph shows top 10 authors who have written maximum books
# plt.figure(1, figsize=(15, 7))
# plt.title("Which aurthor wrote maximum books")
# sns.countplot(x = "authors", order=df['authors'].value_counts().index[0:10] ,data=df)

# #Above Graph shows the most occuring books in the list. Most of these are old all time classics
# plt.figure(1, figsize=(25,7))
# plt.title("Most Occuring Books")
# sns.countplot(x = "title", order=df['title'].value_counts().index[0:10] ,data=df)


# #We infer that English and United States English are the languages most used.
# plt.figure(1, figsize=(25,10))
# plt.title("language_codes")
# sns.countplot(x = "language_code", order=df['language_code'].value_counts().index[0:10] ,data=df)

# #We see that Twilight is the most rated book but it has quite a significant difference in ratings with the other parts.
# most_rated = df.sort_values('ratings_count', ascending = False).head(10).set_index('title')
# sns.barplot(y=most_rated.index, x=most_rated['ratings_count'], palette='Accent')
# plt.figure(figsize=(15,10))


