from nlp_util import word_list_create
import pandas as pd
from gensim import corpora, models
import gensim

WORD_CLASS = ["名詞", "動詞", "形容詞"]
STOP_WORDS = [
    "さん",
    "こと",
    "の",
    "中",
    "時",
    "人",
    "ん",
    "これ",
    "年",
    "わたし",
    "さ",
    "日",
    "月",
    "今日",
    "そう",
    "笑",
]

df = pd.read_csv("./lda_data/master.csv")
sentences = df["tweet"].values.tolist()

docs = word_list_create(sentences, WORD_CLASS, STOP_WORDS)
print(docs[0])

print("辞書の作成")
dictionary = gensim.corpora.Dictionary(docs)
dictionary.save_as_text("./data/text.dict")

print("コーパスの作成")
corpus = [dictionary.doc2bow(doc) for doc in docs]
gensim.corpora.MmCorpus.serialize("./data/text.mm", corpus)

tfidf = gensim.models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

print(" 分類器の学習")
lda = gensim.models.LdaModel(
    corpus=corpus_tfidf,
    id2word=dictionary,
    num_topics=50,
    minimum_probability=0.001,
    passes=20,
    update_every=0,
    chunksize=10000,
)

print(" 分類器の保存")
lda.save("./data/lda.model")
