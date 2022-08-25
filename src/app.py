import streamlit as st
import streamlit.components.v1 as components
import os
import nlplot
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import japanize_matplotlib
from wordcloud import WordCloud
import itertools
import collections
import requests
import datetime

japanize_matplotlib.japanize()

DB_URL = os.environ.get("DB_URL")
engine = create_engine(DB_URL)


@st.cache
def fetch_tweets():
    res = engine.execute('SELECT * FROM tweets').fetchall()
    return [{"index": r[0], "tokens": r[1]
            [1:-1].split(","), "text": r[2], "url": r[3]}for r in res]


@st.cache
def tweet_urls_by_index():
    tweets = fetch_tweets()
    return {t["index"]: t["url"] for t in tweets}


@st.cache
def fetch_tweet_indices():
    res = engine.execute('SELECT * FROM indices').fetchall()
    return [{"index": r[0], "word": r[1], "indices": r[2][1:-1].split(",")}for r in res]


@st.cache
def tweet_indices_by_word():
    indices = fetch_tweet_indices()
    return {i["word"]: i["indices"] for i in indices}


def embed_tweet(url):
    url = f"https://publish.twitter.com/oembed?url={url}"
    res = requests.get(url)
    html = res.json()["html"]
    components.html(html, height=500)


@st.cache
def npt():
    df = pd.DataFrame(fetch_tweets())

    npt = nlplot.NLPlot(df, target_col='tokens', output_file_path="src/")

    stopwords = npt.get_stopword(top_n=0, min_freq=0)
    stopwords += ["midjourney", "Midjourney", "https", "…",
                  ",", "-", "t", "co", "さん", "の", "　", "#", "/", ".", "://", "_", "(", ")", '"', "'", "¥n", "¥n¥n", "@"]

    npt.default_stopwords = stopwords
    return npt


st.set_page_config(layout="wide")
st.title("#midjourney Analysis")

with st.spinner("Loading tweet data..."):
    npt = npt()


@st.cache
def bar_ngram():
    global npt

    return npt.bar_ngram(
        title='上位50単語',
        xaxis_label='単語',
        yaxis_label='頻度',
        ngram=1,
        top_n=200,
        height=3000,
        stopwords=npt.default_stopwords,
    )


@st.cache
def wordcloud():
    global npt

    npt.wordcloud(
        max_words=100,
        max_font_size=100,
        colormap='tab20_r',
        stopwords=npt.default_stopwords,
        save=True
    )
    word_counts = {k: v for k, v in collections.Counter(
        list(itertools.chain.from_iterable(npt.df[npt.target_col].tolist()))).items() if k not in npt.default_stopwords}

    font_path = "src/ipaexg.ttf"
    wordcloud = WordCloud(
        background_color='white',
        stopwords=npt.default_stopwords,
        colormap='tab20_r',
        width=1000,
        height=700,
        font_path=font_path).fit_words(word_counts)
    return wordcloud


@st.cache
def co_network():
    global npt
    npt.build_graph(stopwords=npt.default_stopwords, min_edge_frequency=1)
    npt.co_network(title='network', width=1000, height=700, save=True)

    fname = datetime.datetime.now().strftime("%Y-%m-%d_network.html")
    HtmlFile = open(f"src/{fname}", 'r', encoding='utf-8')
    return HtmlFile.read()


# topwords
st.header("上位単語")
fig = bar_ngram()
st.plotly_chart(fig, use_container_width=True)

# Word Cloud
st.header("ワードクラウド")
wordcloud = wordcloud()
fig, ax = plt.subplots()
plt.imshow(wordcloud, interpolation='bilinear')
st.pyplot(fig)

# CoOccurrence Network
st.header("共起語ネットワーク")
source_code = co_network()
components.html(source_code, width=1000, height=1000)

# TODO: 単語索引（ツイート埋め込み）
st.header("ツイート一覧")
indices = fetch_tweet_indices()
word = st.selectbox("キーワード", [index["word"] for index in indices])

if len(word) > 0:
    tweet_indices = tweet_indices_by_word()[word]
    for i in tweet_indices:
        embed_tweet(tweet_urls_by_index()[int(i)])

# TODO: 画像一覧
