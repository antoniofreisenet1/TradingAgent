#Here we will build a webscraper to gather public opinions of several different stocks.

import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import ace_tools as tools

API_KEY = "9538f324c91b415cb1d968c0a24f05ea"

stock = "AAPL"
sia = SentimentIntensityAnalyzer()

def get_news(stock):
    url = f"https://newsapi.org/v2/everything?q={stock}&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    articles = data["articles"]
    return articles

def analyze_sentiment(articles):
    sentiments = []
    for article in articles:
        sentiment = sia.polarity_scores(article["title"])["compound"]
        sentiments.append(sentiment)
    return sentiments

data = []
stock_info = get_news(stock)
sentiments = analyze_sentiment(stock_info)
for i in range(len(stock_info)):
    data.append([stock_info[i]["title"], sentiments[i]])
    print(stock_info[i]["title"], sentiments[i])
    print()

df = pd.dataFrame(data, columns=["Title", "Sentiment"])
tools.display_df(df)