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

st.set_page_config(page_title="AI Art Twitter Hashtags", layout="wide")

japanize_matplotlib.japanize()

DB_URL = os.environ.get("DB_URL")
engine = create_engine(DB_URL)


@st.cache
def fetch_tweets(service):
    res = engine.execute(f'SELECT * FROM {service}_tweets').fetchall()

    def to_media_urls(text):
        if text == "{}":
            return []
        return text[1:-1].split(",")
    return [{"index": r[0], "tokens": r[1]
            [1:-1].split(","), "text": r[2], "url": r[3], "media_urls": to_media_urls(r[4])}for r in res]


@st.cache
def tweet_urls_by_index(service):
    tweets = fetch_tweets(service)
    return {t["index"]: t["url"] for t in tweets}


@st.cache
def media_urls_by_index(service):
    tweets = fetch_tweets(service)
    return {t["index"]: t["media_urls"] for t in tweets}


@st.cache
def fetch_tweet_indices(service):
    # res = engine.execute(f'SELECT * FROM {service}_indices').fetchall()
    res = engine.execute(
        f"SELECT * FROM {service}_indices ORDER BY Count DESC").fetchall()
    return [{"index": r[0], "word": r[1], "indices": r[2][1:-1].split(",")} for r in res]


@st.cache
def fetch_word_counts(service):
    res = engine.execute(f'SELECT * FROM {service}_wordcount').fetchall()
    return [{"index": r[0], "word": r[1], "count": int(r[2])} for r in res]


@st.cache
def tweet_indices_by_word(service):
    indices = fetch_tweet_indices(service)
    return {i["word"]: i["indices"] for i in indices}


@st.cache
def stopwords(service):
    res = engine.execute(f'SELECT * FROM {service}_stopwords').fetchall()
    return [r[1] for r in res]


def embed_tweet(url):
    url = f"https://publish.twitter.com/oembed?url={url}"
    res = requests.get(url)
    html = res.json()["html"]
    components.html(html, height=500)


def href_image(img_url, href):
    st.markdown(f'''
    <a href="{href}">
        <img src="{img_url}" width="100%"/>
    </a>
    ''', unsafe_allow_html=True)


@st.cache
def bar_ngram(service):
    df = pd.DataFrame(fetch_tweets(service))
    npt = nlplot.NLPlot(df, target_col='tokens', output_file_path="src/")
    return npt.bar_ngram(
        title="",
        xaxis_label='単語',
        yaxis_label='頻度',
        ngram=1,
        top_n=200,
        height=3000,
        stopwords=stopwords(service),
    )


@st.cache
def wordcloud(service):
    df = pd.DataFrame(fetch_tweets(service))
    npt = nlplot.NLPlot(df, target_col='tokens', output_file_path="src/")

    npt.wordcloud(
        max_words=100,
        max_font_size=100,
        colormap='tab20_r',
        stopwords=stopwords(service),
        save=True
    )
    word_counts = {r["word"]: r["count"] for r in fetch_word_counts(service)}

    font_path = "src/ipaexg.ttf"
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords(service),
        colormap='tab20_r',
        width=1000,
        height=700,
        font_path=font_path).fit_words(word_counts)
    return wordcloud


@st.cache
def co_network(service):
    df = pd.DataFrame(fetch_tweets(service))
    npt = nlplot.NLPlot(df, target_col='tokens', output_file_path="src/")

    npt.build_graph(stopwords=stopwords(service), min_edge_frequency=2)
    npt.co_network(title='network', width=1000, height=1000, save=True)

    fname = datetime.datetime.now().strftime("%Y-%m-%d_network.html")
    HtmlFile = open(f"src/{fname}", 'r', encoding='utf-8')
    return HtmlFile.read()


service = st.sidebar.radio(
    "サービス", ["Midjourney", "Stable Diffusion"]).replace(" ", "").lower()
if service not in ["midjourney", "stablediffusion"]:
    st.stop()

opt = st.sidebar.radio("表示", ["画像一覧", "テキスト分析"])

st.markdown(
    f"# [#{service}](https://twitter.com/search?q=%23midjourney&src=recent_search_click) Tweets", unsafe_allow_html=True)

if opt == "画像一覧":
    indices = fetch_tweet_indices(service)
    word = st.selectbox("キーワード", [""] + [
                        index["word"] for index in indices])

    if len(word) > 0:
        cols = st.columns(3)
        tweet_indices = tweet_indices_by_word(service)[word]
        j = 0
        for i in tweet_indices:
            # embed_tweet(tweet_urls_by_index(service)[int(i)])
            urls = media_urls_by_index(service)[int(i)]
            for url in urls:
                col = cols[j % 3]
                with col:
                    href_image(url, tweet_urls_by_index(service)[int(i)])
                    j += 1

elif opt == "テキスト分析":

    # topwords
    st.header("上位単語")
    fig = bar_ngram(service)
    st.plotly_chart(fig, use_container_width=True)

    # Word Cloud
    st.header("ワードクラウド")
    wordcloud = wordcloud(service)
    fig, ax = plt.subplots()
    plt.imshow(wordcloud, interpolation='bilinear')
    st.pyplot(fig)

    # CoOccurrence Network
    st.header("共起語ネットワーク")
    source_code = co_network(service)
    components.html(source_code, width=1000, height=1000)
