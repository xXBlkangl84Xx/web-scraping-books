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