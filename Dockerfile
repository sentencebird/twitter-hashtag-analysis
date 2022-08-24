FROM python:3.7.12

# MeCab
RUN apt-get update
RUN apt-get install -y mecab libmecab-dev mecab-ipadic-utf8 git make curl xz-utils file sudo

# Neologd
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git \
    && cd mecab-ipadic-neologd \
    && bin/install-mecab-ipadic-neologd -n -y

ENV MECABRC="/etc/mecabrc"
ENV MECAB_DIC_DIR="/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd"

COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /usr/src/app

EXPOSE 8501

# CMD ["sh", "-c", "streamlit run --server.port $PORT /usr/src/app.py"]
CMD ["/bin/bash"]