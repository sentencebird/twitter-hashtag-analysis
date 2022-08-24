import MeCab
import os
import tweepy
from sqlalchemy import create_engine
import pandas as pd
import nlplot
import numpy as np

tagger = MeCab.Tagger(
    '/usr/lib/aarch64-linux-gnu/mecab/dic/mecab-ipadic-neologd')

API_KEY = os.environ.get("API_KEY")
API_KEY_SECRET = os.environ.get("API_KEY_SECRET")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.environ.get("ACCESS_TOKEN_SECRET")
DB_URL = os.environ.get("DB_URL")


auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

engine = create_engine(DB_URL)


def parse_tokens(text):
    node = tagger.parseToNode(text)
    d = {}
    while node:
        pos = node.feature.split(',')[0]
        if pos == '名詞' and node.feature.split(",")[1] != "数" or \
                pos == '動詞' or pos == '形容詞':
            d.setdefault(pos, [])
            d[pos].append(node.surface)
        node = node.next
    return d


# Tweet
# res = api.search_tweets(
#     "#midjourney exclude:retweets exclude:replies", lang="ja", count=100)
res = tweepy.Cursor(api.search_tweets, "#midjourney exclude:retweets exclude:replies",
                    lang="ja", count=100).items(1000)
tweet_tokens, texts, urls = [], [], []
for r in res:
    text = r.text
    texts.append(text)
    tweet_tokens.append(parse_tokens(text))
    urls.append(f"https://twitter.com/{r.user.screen_name}/status/{r.id}")
tgt_pos = ["名詞"]

tgt_tokens = [np.hstack([tokens[pos] for pos in tgt_pos if pos in tokens]).tolist(
) for tokens in tweet_tokens]
df = pd.DataFrame({"tokens": tgt_tokens, "texts": texts, "urls": urls})

df.to_sql("tweets", con=engine, if_exists="replace")


# Word Index
npt = nlplot.NLPlot(df, target_col='tokens')

stopwords = npt.get_stopword(top_n=0, min_freq=0)
stopwords += ["midjourney", "Midjourney", "https", "…",
              ",", "-", "t", "co", "さん", "の", "　", "#", "/", ".", "://", "_", "(", ")", '"', "'", "¥n", "¥n¥n", "@"]
ngram_df = npt.df.copy()
ngram_df.loc[:, 'space'] = ngram_df[npt.target_col].apply(
    lambda x: ' '.join(x))

topwords = nlplot.nlplot.freq_df(ngram_df["space"], stopwords=stopwords, n=200)[
    "word"].tolist()

indices_by_topword = {}
for i, row in npt.df.iterrows():
    words = set(row["tokens"]) & set(topwords)
    for word in words:
        indices_by_topword.setdefault(word, [])
        indices_by_topword[word].append(i)

indices_df = pd.DataFrame(indices_by_topword.items(),
                          columns=["Word", "Indices"])
indices_df.to_sql("indices", con=engine, if_exists="replace")
