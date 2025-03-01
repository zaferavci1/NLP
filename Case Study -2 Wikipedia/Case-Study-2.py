import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from textblob import Word

df = pd.read_csv("Case Study -2 Wikipedia/wiki_data.csv")
df.columns = ["id", "text"]
df.head()

def clean_text(data):
    data = str(data).lower()
    data = re.sub(r"[^\w\s]", " ", data)
    data = re.sub(r"\d", " ", data)
    return data

df["text"] = df["text"].apply(clean_text)

rando = df["text"].sample()
rando
a = clean_text(rando)
print(a)

sw = stopwords.words("english")

def remove_stopwords(data):
    data = " ".join(word for word in str(data).split() if word not in sw)
    return data

df["text"] = df["text"].apply(lambda x: remove_stopwords(x))

text = pd.Series(" ".join(df["text"]).split()).value_counts()
text.head()

drops = text[text <= 1]

df["text"] = df["text"].apply(lambda x: " ".join(word for word in str(x).split() if word not in drops))

df["text"] = df["text"].apply(lambda x: " ".join([Word(a).lemmatize() for a in str(x).split()]))