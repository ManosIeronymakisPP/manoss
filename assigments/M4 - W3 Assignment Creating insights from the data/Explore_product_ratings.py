# importing the required libraries.
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


#import goodreads data, skipping a few corrupted rows (extra commas)
os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad")
filepath =  "books.csv"
full_data = pd.read_csv(filepath, error_bad_lines=False)

# Showing a sample of the data
cm = sns.light_palette("lightblue", as_cmap=True)
full_data.head(10).style.background_gradient(cmap=cm)

#################################### EDA ####################################################################################

#The "/" signs in the authors' column can cause problems when answering questions like "Who's the highest rated author?".
#To avoid these issues, let's only consider the first author of each book with two authors, even though it may be unfair to some collaborators. This will improve the clarity and accuracy of our analysis.
# a function that count how many books have more than one auther.
def slash_in_data(col,sign='/'):
    n=0
    for s in col:
        if sign in s:
            n+=1
    return n

slash_in_data(full_data.authors)

# a loop to specify only the first author in each book that has more than one author.
for i,s in enumerate(full_data.authors):
    if '/' in s:
        slash_i = s.index('/')
        full_data.authors.iloc[i] = s[:slash_i]

slash_in_data(full_data.authors) # checking again if there are still a more than one author books in our data.

# showing a statistical review about the data.
full_data.describe().style.background_gradient(cmap=cm)

#Upon a quick statistical review, I've noticed zero values in the num_pages and ratings_count columns that may indicate unimportant books. 
#Additionally, there are clear outliers in the num_pages, ratings_count, and text_reviews_count columns.
#While high ratings_count and text_reviews_count may indicate a popular book, a high num_pages value does not always correlate with popularity. 
#To improve our analysis, we'll remove books with zero values and outliers.
full_data.rename(columns={'  num_pages':'num_pages'},inplace=True) # renaming the num_pages colums to be striped.
full_data.plot(kind='scatter',x='bookID',y='num_pages',figsize=(20,12)); #ploting a scatter plot to show how bad the outliers are.

# From the last scatter plot I can see that there are unacceptable outliers that are above the value of 2 thousand.
#filtering our data 
full_data = full_data[(20<full_data.num_pages)&(full_data.ratings_count > 100)] # geting only the books with num_pages value greater than 20 and ratings_count greater than 100.
full_data.describe().style.background_gradient(cmap=cm)
full_data.plot(kind='scatter',x='bookID',y='num_pages',figsize=(20,12));

# geting the number of null values that occured after removing the outliers.
full_data.isnull().sum().sum()

#getting only the data we'll need in our Analysis
data = full_data[['title','authors','average_rating','language_code','num_pages','ratings_count','text_reviews_count','publication_date','publisher']]
data.head(10).style.background_gradient(cmap=cm)


#############################  What are the top 10 popular books ever? #####################################################


#Sort the data by average rating in descending order and select the top 10 books. Apply a color gradient to highlight the top-rated books.
data.sort_values('average_rating',ascending = False).head(10).style.background_gradient(cmap=cm)

#Calculate a weighted score for each book based on the number of ratings and the average rating. Assign this score to a new column called 'real_best_books'. 
#Show the top 10 books with the highest score and apply a color gradient to highlight them.
data['real_best_books'] = (data.average_rating**8 * data.ratings_count)/10**5
data.head(10).style.background_gradient(cmap=cm)

#Sort the data by the 'real_best_books' column in descending order and select the top 10 books. Apply a color gradient to highlight them.
most_rated_books = data.sort_values('real_best_books', ascending=False).head(10)
most_rated_books.style.background_gradient(cmap=cm)

#Create a horizontal bar chart to visualize the top 10 books with the highest 'real_best_books' score.
plt.figure(figsize=(15,10))
sns.barplot(x=most_rated_books['real_best_books'], y=most_rated_books.title, palette='cool');




###################################  Who are the top 10 popular authors?  ##################################################################################

# Display the top 10 authors by real best books score
pd.DataFrame(data=data.sort_values('real_best_books', ascending=False)['authors'].unique()[:10], columns=['top_authors'])

# Group data by authors and calculate the mean of their average rating and ratings count
top_authors = data[['authors','average_rating','ratings_count']].groupby('authors').mean()

# Calculate the real score for each author based on their average rating and ratings count
top_authors['real_score'] = (top_authors.average_rating**8 * top_authors.ratings_count)/10**5

# Display the top 10 authors by real score
top_10_authors = top_authors.sort_values('real_score', ascending=False).head(10)
top_10_authors.style.background_gradient(cmap=cm)

# Create a bar plot to visualize the top 10 authors by vote score
plt.figure(figsize=(15,10))
ax = sns.barplot(x= top_10_authors.real_score,y=  top_10_authors.index, palette='cool')
ax.set_title("Top 10 voted authors")
ax.set_xlabel("Vote score")
for i in ax.patches:
    ax.text(i.get_width()+.3, i.get_y()+0.5, str(round(i.get_width())), fontsize = 10, color = 'k')


############################### Who are the top 10 authors the highest number of books? ###################################################################

# Select the top 10 authors with the highest number of books
most_writing_authors = data.authors.value_counts().head(10).to_frame()

# Apply background gradient to style the table
most_writing_authors.style.background_gradient(cmap=cm)

# Create a barplot to visualize the top 10 authors with the highest number of books
plt.figure(figsize=(15,10))
ax = sns.barplot(x = most_writing_authors.authors,y = most_writing_authors.index, palette='cool')
ax.set_title("Top 10 authors with the highest number of books")
ax.set_xlabel("Total number of books")

# Add labels to the barplot
for i in ax.patches:
    ax.text(i.get_width()+.3, i.get_y()+0.5, str(round(i.get_width())), fontsize = 10, color = 'k')


###################################### What the most frequent language in the data? ########################################################################## 

data.language_code.value_counts().head(10).to_frame().style.background_gradient(cmap=cm)
plt.figure(figsize=(15,10))
ax = data.groupby('language_code')['title'].count().plot.bar()
plt.title('Language Code')
plt.xticks(fontsize = 15)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x()-0.3, p.get_height()+100))

################################## What are the average arting for each language? ########################################################################

plt.figure(figsize=(15,10))
ax = data.groupby('language_code')['average_rating'].mean().sort_values(ascending=False).plot.bar()
plt.title('Language Code')
plt.xticks(fontsize = 15)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x()-0.3, p.get_height()+100))

##############################  What are the top 10 largest books ever? #####################################################

data.sort_values('num_pages',ascending=False).head(10).style.background_gradient(cmap=cm)

############################################# Who is the top 10 authors with the largest books ever? #################################################

most_bigbook_authors = data[['authors','num_pages']].groupby('authors').mean().sort_values('num_pages',ascending=False).head(10)
most_bigbook_authors.style.background_gradient(cmap=cm)
plt.figure(figsize=(15,10))
sns.barplot(x= most_bigbook_authors.num_pages, y= most_bigbook_authors.index, palette='cool');

#############################  Is there's a relationship between the number of pages and the average rarting? #############################################

plt.figure(figsize=(180,90))
sns.lmplot(x="num_pages", y="average_rating", data=data);

#Well, it's a weak relationship but still a positive relationship between the number of pages and average rating.

############################  What are the top 10 languages with the kargest books ever? #####################################################

most_bigbook_langs = data[['language_code','num_pages']].groupby('language_code').mean().sort_values('num_pages',ascending=False).head(10)
most_bigbook_langs.style.background_gradient(cmap=cm)

plt.figure(figsize=(15,10))
sns.barplot(x =most_bigbook_langs.num_pages,y =  most_bigbook_langs.index, palette='cool');

########################################  What are the top 10 read books ever? ##########################################################
#we are considering that the rating count represents the number of reads.

data.sort_values('ratings_count',ascending=False).head(10).style.background_gradient(cmap=cm)

###################################  Who are the top 10 authors with most read books ever? ###############################################

top_read_authors = data[['authors','ratings_count']].groupby('authors').mean().sort_values('ratings_count',ascending=False).head(10)
plt.figure(figsize=(15,10))
sns.barplot(x = top_read_authors.ratings_count,y=  top_read_authors.index, palette='cool');

###################################### Is there's a relationship between the number of reviews and the average rate? ###########################################################
plt.figure(figsize=(180,90))
sns.lmplot(x="average_rating", y="ratings_count", data=data);

# ## The answer seems to be NO, although all of the books with rating counts the value of more than 1 million have an average rating of more than 3.5

############################### Is there's a relationshop between the rating counts and number of bages? ####################################################################

plt.figure(figsize=(180,90))
sns.lmplot(x="num_pages", y="ratings_count", data=data);
#Once again, the answer is NO, although all of the books with ratings more than 1 million have number of pages under 1000 pages. 

############################### What are the most books with the highest text reviews count? ###################################################################

most_text_r = data.sort_values('text_reviews_count',ascending=False).head(10)
most_text_r.style.background_gradient(cmap=cm)
plt.figure(figsize=(15,10))
sns.barplot(x = most_text_r.text_reviews_count,y=  most_text_r.title, palette='cool');

#######################################  Who are the top 10 authors with the highst text reviews count? ######################################################
most_author_text = data[['authors','text_reviews_count']].groupby('authors').mean().sort_values('text_reviews_count',ascending=False).head(10)
most_author_text.style.background_gradient(cmap=cm)
plt.figure(figsize=(15,10))
sns.barplot(x= most_author_text.text_reviews_count,y= most_author_text.index, palette='cool');

###########################  Is there's a relationship between the text reviews count and the average rating? ####################################################

plt.figure(figsize=(180,90))
sns.lmplot(x="average_rating", y="text_reviews_count", data=data);

#The answer is NO, although all of the books that have a text reviews count values more than 20 thousand also have an average rating values that are more than 3.5.

################################### Is there's a relationship between the text reviews count and ratings count? ##################################################
plt.figure(figsize=(180,90))
sns.lmplot(x="text_reviews_count", y="ratings_count", data=data);

#the answer is yes, there's a strong positive relationship between text reviews count and ratings count (though it's very obvious).

#################################### How does the average vote goes through the years? ###################################################################
data.publication_date = pd.to_datetime(data.publication_date,errors='coerce')
data['year'] = data.publication_date.dt.year
year_avg_rate = data[['year','average_rating']].groupby('year').mean()['average_rating']

plt.figure(figsize=(25,10))
ax = year_avg_rate.plot()
plt.title('Language Code')
plt.xticks(fontsize = 15)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x()-0.3, p.get_height()+100))

##################################  What is the most voted book for each single year in the data? ###################################################
yy = data[['year','average_rating']].groupby(['year']).max()
dct = dict()
for rate,y in zip(yy.average_rating,yy.index):
    dct[y] = [data.authors[(data.year == y)&(data.average_rating == rate)].iloc[0],data.title[(data.year == y)&(data.average_rating == rate)].iloc[0],rate]
dct

best_book_author_y = pd.DataFrame(data= dct.values(), columns = ['author','book','rate'],index=dct.keys())
best_book_author_y.style.background_gradient(cmap=cm)

#####################################  How does the rating count (the reads) goes through the years? ####################################################################

year_count_rate = data[['year','ratings_count']].groupby('year').mean()['ratings_count']
plt.figure(figsize=(25,10))
ax = year_count_rate.plot()
plt.title('Language Code')
plt.xticks(fontsize = 15)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x()-0.3, p.get_height()+100))

#The data seems to have som extreme outlier that is between 1950-1960.

data[(data.year > 1950) & (data.year < 1960)].sort_values('ratings_count',ascending= False).head().style.background_gradient(cmap=cm)

# ## From the data above, the outlier seems to be a book called "A Streetcar Named Desire" that is the only book that have been published in 1952 with a ratings count equals to 235224.

for i,s in enumerate(data.publisher):
    if '/' in s:
        slash_i = s.index('/')
        data.publisher.iloc[i] = s[:slash_i]

#####################################################  Who are the top 10 popular publishers? ####################################################################

d_r_p = data[['real_best_books','publisher']].groupby('publisher').mean().sort_values('real_best_books',ascending=False).head(10)
plt.figure(figsize=(15,10))
sns.barplot(x= d_r_p.real_best_books,y= d_r_p.index, palette='cool');

#Observations:
    
#The most popular books are Harry Potter's books.
#The most popular author is Stephenie Meyer.
#Latin books have the highest average average_ratings.
#All of the books with a number of pages that is more then 1500 pages have an average rarings more than 4.
#Twilight is probably the most read book ever.
#All of the books with rating counts the value of more than 1 million have an average rating of more than 3.5.
#All of the books with ratings more than 1 million have number of pages under 1000 pages.
#All of the books that have a text reviews count values more than 20 thousand also have an average rating values that are more than 3.5.
#Arthur A. Levine Books are is the most popular publishing company.


################################################## What distribution do you think the ''average_rating" has? #################################################################

sns.kdeplot(data=data, x='average_rating')

#the distribution look like to be normal


############################## correlation matrix of the features. ###########################

# Select the columns of interest
cols = ['average_rating', 'num_pages', 'ratings_count', 'text_reviews_count']

# Create the correlation matrix
corr = full_data[cols].corr()

# Plot the correlation matrix as a heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Book Features')
plt.show()










