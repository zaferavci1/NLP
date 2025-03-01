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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_excel("Case Study -1 Amazon/amazon.xlsx")
df.columns = ["star", "helpful", "title", "review"]

df.head()

df["review"]

df["review"] = df["review"].str.lower()
df.head()

df["review"] = df["review"].str.replace(r"[^\w\s]", " ", regex=True)

df["review"] = df["review"].str.replace(r"\d", " ", regex=True)

import nltk
#nltk.download('stopwords')
sw = stopwords.words("english")

df["reviewText"] = df["review"].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

temp_df = pd.Series(" ".join(df["reviewText"]).split()).value_counts()
drops = temp_df[temp_df <=1]

df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(a for a in str(x).split() if a not in drops))

#lemmatization

df["reviewText"] = df["reviewText"].apply(lambda x: " ".join([Word(a).lemmatize() for a in str(x).split()]))

tf = df["reviewText"].apply(lambda x: pd.Series(str(x).split()).value_counts()).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]

tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

text = " ".join(a for a in df["reviewText"])
text
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# sentiment analysis

#nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()
sia.polarity_scores("the film is awesome")

df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["polarity_scores"] = df["reviewText"].apply(lambda x: sia.polarity_scores(x)["compound"])
df.columns

df["polarity_scores"][0:10].apply(lambda x: f"{x} pos" if x > 0 else f"{x} neg")

df["sentiment"] = df["polarity_scores"].apply(lambda x: "pos" if x>0 else "neg")

df["sentiment"].value_counts()

df.groupby("sentiment").agg({"star":"mean"})

df["sentiment"] = LabelEncoder().fit_transform(df["sentiment"])

X = df["reviewText"]
y = df["sentiment"]

tf_idf = TfidfVectorizer()

X_tf_idf_word = TfidfVectorizer().fit_transform(X)

log_model = LogisticRegression().fit(X_tf_idf_word, y)

new_comment = pd.Series("the item is great")

new_comment = TfidfVectorizer().fit(X).transform(new_comment)

log_model.predict(new_comment)

cross_val_score(log_model,X_tf_idf_word, y, cv=5, scoring="accuracy", n_jobs=-1).mean()

df["reviewText"].sample()

vectorizer = CountVectorizer()
X_count_vec = vectorizer.fit_transform(X)

random_review = pd.Series(df["reviewText"].sample())

random_review = vectorizer.transform(random_review)

random_review

print(log_model.predict(random_review))
