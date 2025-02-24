from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

df = pd.read_csv("amazon_reviews.csv", sep=",")

df.head()   

df["reviewText"] = df["reviewText"].str.lower()
df["reviewText"]

#Punctuations
df["reviewText"] = df["reviewText"].str.replace(r'[^\w\s]', " ", regex=True)
df["reviewText"]

#Numbers
df["reviewText"] = df["reviewText"].str.replace(r'\d', " ", regex=True)
df["reviewText"] 

#Stopwords
import nltk
#nltk.download('stopwords')
sw = stopwords.words("english")

df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

#Rare words
temp_df = pd.Series(" ".join(df["reviewText"]).split()).value_counts()

drops = temp_df[temp_df <= 1]

df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))

#Tokenization
#nltk.download("punkt_tab")

df["reviewText"].apply(lambda x: TextBlob(x).words).head()

#Lemmatization 
# kelimeleri köklerine ayırma işlemi
#nltk.download("wordnet")