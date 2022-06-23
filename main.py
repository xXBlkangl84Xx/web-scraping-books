import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
import re

import csv as csv
from datetime import datetime
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats

import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import LabelEncoder

import re
import string
import gensim
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')


import requests
from bs4 import BeautifulSoup
import lxml


# formatting
pd.options.display.max_colwidth = 100
pd.options.display.float_format = '{:,.2f}'.format

# make a get request to the books overview page
url = 'https://books.toscrape.com/catalogue/category/books_1/index.html'
res = requests.get(url)
soup = BeautifulSoup(res.text, "lxml")

# create a BeautifulSoup object containing every book preview. Return the href attribute in the <a> tag nested within the first product class element
soup.find_all(class_="product_pod")[0].find("a").attrs["href"]

## FORMAT AND CONCATENATE TO FULL URL
# add the prefix "https://books.toscrape.com/catalogue/" and combine with href to get full link
baselink = "https://books.toscrape.com/catalogue/"
pd.Series(soup.find_all(class_="product_pod")).apply(lambda x : baselink + x.h3.find("a")["href"].split("/")[2]+"/index.html").head()

# find book titles by finding <article> tags with class = product_pod, access index 0 and find <a> tag attribute "title" which is nested in an <h3> tag
soup.findAll('article' , {'class' , "product_pod"})[0].h3.find('a')['title']

# find book prices by finding <article> tags with class = product_pod, access index 0 and find <div> tag with class = product_price and then nested <p> tag, which contains the price. Then strip the text, encode it as ascii (ignoring symbols i.e. '$'), and then decode as ascii

soup.findAll('article' , {'class' , "product_pod"})[0].find('div' , {'class' , "product_price"}).p.text.strip().encode("ascii" , "ignore").decode("ascii")

## FINDING HOW MANY PAGES OF BOOKS THERE ARE
count = 1
# create BeautifulSoup object with the html contents of every book page
sp = BeautifulSoup(requests.get(baselink + "page-{}.html".format(count)).text , 'html.parser')

# loop through BeautifulSoup object 'sp' until it encounters the text "Not Found"
while sp.text.find("Not Found") == -1:
    count += 1
    sp = BeautifulSoup(requests.get(baselink + "page-{}.html".format(count)).text , 'html.parser')

## CREATE FUNCTION TO ITERATE OVER EVERY BOOK ON EVERY PAGE AND SCRAPE RELEVANT INFORMATION
def get_book_data(sublink):
    link = baselink + sublink
    res = requests.get(link)
    soup = BeautifulSoup(res.text , 'html.parser')
    # scrape book description
    description = soup.findAll("meta" , {'name' : "description"})[0].attrs['content']
    # scrape book rating
    rating = soup.select(".star-rating")[0]['class'][1]
    # scrape book name
    name = soup.find("div" , {'class' : "col-sm-6 product_main"}).h1.text
    # scrape book price
    price = soup.find("div" , {'class' : "col-sm-6 product_main"}).select("p")[0].text
    # scrape book topic
    topic = soup.find("ul" , {'class' : "breadcrumb"}).findAll("a")[2].text
    # scrape book stock number and then availability status
    stock = soup.find("div" , {'class' : "col-sm-6 product_main"}).select("p")[1].text.replace("\n","").strip()
    try : items = re.findall("\d+",stock)[0]
    except IndexError : item = 0

    try :
        availability = stock.split("(")[0]
    except SyntaxError : availability = stock

    availability = availability.strip()
    # scrape remaining book information from table
    table_values = []
    for i in soup.findAll("table")[0].select("tr"):
        table_values.append(i.select("td")[0].text)

    UPC , product_type, reviews = table_values[0], table_values[1], table_values[6]

    return [UPC, name, product_type, price, availability, items, topic, rating, reviews, description]

## CREATE LOOP TO ITERATE OVER PAGES AND OVER BOOKS WITHIN EACH PAGE
books_data = []
for count in range(1, count):
    # scraping from each book page
    res1 = requests.get(baselink + "page-{}.html".format(count))
    soup1 = BeautifulSoup(res1.text, 'html.parser')
    books_links = soup1.findAll("section")[0].select('h3')
    for i in books_links:
        sublink = i.a['href']
        books_data.append(get_book_data(sublink))

# create dataframe of every book and it's scraped information
#books_df = pd.DataFrame(books_data, columns = ["UPC", "name", "product_type", "price", "availability", "items", "topic", "rating", "reviews", "description"])

# export books dataframe to a csv file
#books_df.to_csv("books.csv")

# read from and output books.csv
books_df = pd.read_csv("books.csv")
# books_df.head()

## FIX ISSUES WITH AND CLEAN DATA

# removing columns product_type, availability, and reviews since every book has the same value for each
books_df.drop(columns = ["product_type", "reviews", "availability"] , inplace=True)

# stripping non-ASCII characters from price and converting to float type
books_df["price"] = books_df["price"].apply(lambda x : re.findall("\d+.\d+", x)[0])
books_df["price"] = books_df["price"].astype("float")

#  changing items count and rating to type float
rating_dic = {"One":1, "Two":2, "Three":3, "Four":4, "Five":5}
books_df["rating"] = books_df["rating"].map(rating_dic)
books_df["rating"] = books_df["rating"].astype("float")

books_df["items"] = books_df["items"].astype("float")

## NLP PROCESSING FOR DESCRIPTION AND TOPIC
# create function for text cleaning and vectorization
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    #text = re.sub('’', '', text)
    text = re.sub('\w*\d\w*', '', text) # drop unneeded characters
    tokens = nltk.word_tokenize(text) # tokenizing words
    tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]# removing stopwods
    tokens = [lemmatizer.lemmatize(w) for w in tokens] # lemmatizing tokenized words
    text = " ".join(tokens)
    text = text.lower().strip()
    return text

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

# instantiate vectorizer
vect = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95, lowercase=False, max_features=100000, stop_words='english')

# create class with functions to get most frequent words in each cluster of texts
class TfidVectorizer :
    def __init(self, x):
        self.x = x
    def fit(self):
        return vect.fit_transform(self).toarray()
    def transform(self):
        return vect.transform(X_splitted)

# create function that draws an elbow graph to determine best number of clusters for a sample of data using K-means clustering
def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax+1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0
      # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(points.shape[0]):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
      
        sse.append((k , curr_sse))
    return sse

## CLUSTERING BOOK DESCRIPTIONS
# clean text from NAN values
text = books_df["description"].replace(np.nan, " ").astype(str)
# pass text to clean_text function. Making text lowercase and removing text in brackets, punctuation, links, and words with numbers
clean_text(text[1])
text = text.apply(lambda x : clean_text(x))

# vectorize text using tfidf vectorizer
X = vect.fit_transform(text)

# use K-means to cluster text after getting best number of clusters from elbow technique
rss = calculate_WSS(X, 15)
df = pd.DataFrame(rss)
plt.figure(figsize=(16,4))

# plot kmeans
# g = sns.lineplot(data=df, x=0, y=1)
# g.set(xticks=df[0])
# plt.show()

# best number of clusters (from above graph) is 8. Initialize kmeans with 8 centroids
kmeans = KMeans(n_clusters=8, random_state=42)
# fit the model
kmeans.fit(X)
# store cluster labels (book topics) in a variable
clusters = kmeans.labels_

books_df['book_topic'] = clusters+1

## MERGING TOPIC CATEGORIES TO EASE PREDICTION
# create dictionary and map onto topics, changing the topics
topic_dict={	'Academic'	:	'Cultural'	,
            'Adult Fiction'	:	 'Light Arts'
            	, 'Art'	:	 'Art'	, 
            'Autobiography'	:	 'social sciences'
            	, 'Biography'	:	 'social sciences'	, 
            'Business'	:	 'Cultural'	,
            'Childrens'	:	 'Light Arts'	
            , 'Christian'	:	 'Spirituality'
            	, 'Christian Fiction'	:	 'Spirituality'
              	, 'Classics'	:	 'Art'	, 
            'Contemporary'	:	 'Art'	, 
            'Crime'	:	 'social sciences'	, 
            'Cultural'	:	 'Cultural'
            	, 'Erotica'	:	 'Light Arts'	,
             'Fantasy'	:	 'Light Arts'	,
             'Fiction'	:	 'Light Arts'	
            , 'Food and Drink'	:	 'Light Arts'	,
            'Health'	:	 'Cultural'	, 
            'Historical'	:	 'social sciences'	,
            'Historical Fiction'	:	 'social sciences'	,
            'History'	:	 'social sciences'	,
            'Horror'	:	 'Art'	,
            'Humor'	:	 'Art'	,
            'Music'	:	 'Art'	
            , 'Mystery'	:	 'Art'
            	, 'New Adult'	:	 'Light Arts'
              	, 'Nonfiction'	:	 'Nonfiction'	
            , 'Novels'	:	 'Art'	,
            'Paranormal'	:	 'Art'
            	, 'Parenting'	:	 'social sciences'
              	, 'Philosophy'	:	 'social sciences'	
            , 'Poetry'	:	 'Art'
            	, 'Politics'	:	 'social sciences'	
            , 'Psychology'	:	 'social sciences'	
            , 'Religion'	:	 'Spirituality'	
            , 'Romance'	:	 'Art'	
            , 'Science'	:	 'Cultural'
            	, 'Science Fiction'	:	 'Cultural'	
            , 'Self Help'	:	 'social sciences'
            	, 'Sequential Art'	:	 'Sequential Art'
              	, 'Short Stories'	:	 'Art'
                	, 'Spirituality'	:	 'Spirituality'	
            , 'Sports and Games'	:	 'Light Arts'	
            , 'Suspense'	:	 'Art'	
            , 'Thriller'	:	 'Art'	, 
            'Travel'	:	 'Art'	
            , 'Womens Fiction'	:	 'Light Arts'
            	, 'Young Adult'	:	 'Light Arts'	,
             'Default'	:	 'Default' ,
            'Add a comment'	:	 'Add a comment' ,
             'Nonfiction'	:	 'Nonfiction'}

books_df["topic"] = books_df["topic"].map(topic_dict)

# create function to replace neutral topics using new clusters


def get_top_keywords(n_terms):
    """This function returns the keywords for each centroid of the KMeans"""
    df = pd.DataFrame(X.todense()).groupby(clusters).mean() # groups the TF-IDF vector by cluster
    terms = vect.get_feature_names_out() # access tf-idf terms
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i+1))
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) # for each row of the dataframe, find the n terms that have the highest tf idf score
        try : print(   books_df.loc[books_df['book_topic']==i+1 ,['topic']].value_counts(normalize=True).head(10)*100)
        
        except KeyError : pass
        try : new_topics[i+1] =  books_df.loc[books_df['book_topic']==i+1 ,['topic']].value_counts(normalize=True).index.drop(["Default" ,"Add a comment"  , "Nonfiction"])[0][0]
        except KeyError : pass    

# empty dictionary for new topics
new_topics = {}

# call get_top_keywords function with value 10 to get the 10 most common topics in description
#get_top_keywords(10)

# map new topics dictionary onto dataframe topic column
books_df.loc[(books_df['topic'] == 'Default') | (books_df['topic'] == 'Add a comment') | (books_df['topic'] == 'Nonfiction') , ["topic"]] = books_df["book_topic"].map(new_topics)

books_df.drop(columns = ["description"] , inplace = True)

books_df["book_topic"] = books_df["book_topic"].astype("object")


## PERFORMING EXPLORATORY DATA ANALYSIS ON FEATURE OF INTEREST "TOPIC"

books_df.groupby("topic").mean()['price'].plot(kind='barh')