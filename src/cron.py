import MeCab
import os
import tweepy
from sqlalchemy import create_engine
import pandas as pd
import nlplot
import numpy as np
import string
from tqdm import tqdm
import argparse
import collections
import itertools

parser = argparse.ArgumentParser()

parser.add_argument('-s', default="midjourney")
args = parser.parse_args()
service = args.s

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

default_stopwords = ["midjourney", "Midjourney", "midjourneyart", "midjourneyAi", "ミッドジャーニー", "stablediffusion",
                     "AI", "画像", "絵", "beta", "aiart", "of", "the", "in",
                     "https", "…", ",", "-",  "co", "さん", "の", "　", "#", "/", ".", "://", "_", "(", ")", '"', "'", "¥n", "¥n¥n", "@"] + [s for s in string.ascii_letters]


def parse_tokens(text, stopwords=default_stopwords):
    node = tagger.parseToNode(text)
    d = {}
    while node:
        pos = node.feature.split(',')[0]
        if pos == '名詞' and node.feature.split(",")[1] in ["一般", "固有名詞"] and node.surface not in stopwords:
            d.setdefault(pos, [])
            d[pos].append(node.surface)
        node = node.next
    return d


# Tweet
res = tweepy.Cursor(api.search_tweets, f"#{service} exclude:retweets exclude:replies",
                    lang="ja", count=100).items(5000)
tweet_tokens, texts, urls, media_urls = [], [], [], []
for r in tqdm(res):
    text = r.text
    texts.append(text)
    tweet_tokens.append(parse_tokens(text))
    urls.append(f"https://twitter.com/{r.user.screen_name}/status/{r.id}")
    media_urls.append([m["media_url"] for m in r.extended_entities['media']] if hasattr(
        r, "extended_entities") else [])
tgt_pos = ["名詞"]

tgt_tokens = [np.hstack([tokens[pos] for pos in tgt_pos if pos in tokens]).tolist(
) if tokens != {} else [] for tokens in tweet_tokens]
df = pd.DataFrame({"tokens": tgt_tokens, "texts": texts,
                  "urls": urls, "media_urls": media_urls})

df.to_sql(f"{service}_tweets", con=engine, if_exists="replace")

# Word Index
npt = nlplot.NLPlot(df, target_col='tokens')

stopwords = npt.get_stopword(top_n=20)
stopwords += default_stopwords

pd.DataFrame(stopwords, columns=["words"]).to_sql(
    "stopwords", con=engine, if_exists="replace")

ngram_df = npt.df.copy()
ngram_df.loc[:, 'space'] = ngram_df[npt.target_col].apply(
    lambda x: ' '.join(x))

topwords = nlplot.nlplot.freq_df(ngram_df["space"], stopwords=stopwords, n=1000)[
    "word"].tolist()

indices_by_topword = {}
for i, row in npt.df.iterrows():
    words = set(row["tokens"]) & set(topwords)
    for word in words:
        indices_by_topword.setdefault(word, [])
        indices_by_topword[word].append(i)

d = [{"word": word, "tweet_indices": indices, "count": len(
    indices)} for word, indices in indices_by_topword.items()]
indices_df = pd.DataFrame(d)
indices_df.to_sql(f"{service}_indices", con=engine, if_exists="replace")

# Word Cloud
word_counts = {k: v for k, v in collections.Counter(
    list(itertools.chain.from_iterable(npt.df[npt.target_col].tolist()))).items() if k not in stopwords}
pd.DataFrame(word_counts.items(), columns=["word", "count"]).to_sql(
    f"{service}_wordcount", con=engine, if_exists="replace")
