import gensim
from dataclasses import dataclass
from nlp_util import word_list_create

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


@dataclass
class LdaVectorize:
    lda: gensim
    dictionary: gensim

    def execute(self, text):
        dic_vec = self.dictionary.doc2bow(
            word_list_create([text], WORD_CLASS, STOP_WORDS)[0]
        )
        lda_vec = self.lda[dic_vec]
        result = []
        for lv in lda_vec:
            result.append(lv[1])
        return result
