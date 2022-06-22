import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import svm

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

# FORMATTING
pd.options.display.max_colwidth = 100
pd.options.display.float_format = '{:,.2f}'.format

# SCRAPING - make a get request to the books overview page
url = 'https://books.toscrape.com/catalogue/category/books_1/index.html'
res = requests.get(url)
soup = BeautifulSoup(res.text, "lxml")

# create a BeautifulSoup object containing every book preview. Return the href attribute in the <a> tag nested within the first product class element
soup.find_all(class_="product_pod")[0].find("a").attrs["href"]

## format the url to make it complete
# add the prefix "https://books.toscrape.com/catalogue/" and combine with href to get full link
baselink = "https://books.toscrape.com/catalogue/"
pd.Series(soup.find_all(class_="product_pod")).apply(lambda x : baselink + x.h3.find("a")["href"].split("/")[2]+"/index.html").head()

# find book titles by finding <article> tags with class = product_pod, access index 0 and find <a> tag attribute "title" which is nested in an <h3> tag
soup.findAll('article' , {'class' , "product_pod"})[0].h3.find('a')['title']

# find book prices by finding <article> tags with class = product_pod, access index 0 and find <div> tag with class = product_price and then nested <p> tag, which contains the price. Then strip the text, encode it as ascii (ignoring symbols i.e. '$'), and then decode as ascii

soup.findAll('article' , {'class' , "product_pod"})[0].find('div' , {'class' , "product_price"}).p.text.strip().encode("ascii" , "ignore").decode("ascii")

# loop through BeautifulSoup object containing book previews from first page and then fill in the empty book_urls array 
# for i in soup.findAll('article' , {'class' , 'product_pod'}) :
#     print(i.h3.find('a')['title'])
#     print('Price: ' + i.find('div' , {'class' , "product_price"}).p.text.strip().encode("ascii" , "ignore").decode("ascii"))
#     print("Availability : "+( i.find('p' , {'class' , "instock availability"}).text.replace(" " ,"")).strip())
#     print("Class Rating : "+ i.find('p' , {'class' , "star-rating"})['class'][1])
#     print("Link of Books  Data : " + baselink + i.h3.find("a")["href"].split("/")[2]+"/index.html" )
#     print(' ')
#     print(' *****************')

## Finding how many pages of books there are
count = 1
# create BeautifulSoup object with the html contents of every book page
sp = BeautifulSoup(requests.get(baselink + "page-{}.html".format(count)).text , 'html.parser')

# loop through BeautifulSoup object 'sp' until it encounters the text "Not Found"
while sp.text.find("Not Found") == -1:
    count += 1
    sp = BeautifulSoup(requests.get(baselink + "page-{}.html".format(count)).text , 'html.parser')

## Create function to iterate over every book on every page and scrape relevant information
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

## Create loop to iterate over pages and over books within each page
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
books_df = pd.DataFrame(books_data, columns = ["UPC", "name", "product_type", "price", "availability", "items", "topic", "rating", "reviews", "description"])

# export books dataframe to a csv file
books_df.to_csv("books.csv")

# read from and output books.csv
books_df = pd.read_csv("books.csv")
books_df.head()