import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from util_m import CosSimilar
from nlp_util import word_list_create
from tfidf_vectorize import TfidfVectorize
from dataclasses import dataclass
import MeCab

WORD_CLASS = ["名詞", "動詞", "形容詞"]
STOP_WORDS = []


@dataclass
class TfidfCosCalc:
    tfidf_vectorize: TfidfVectorize
    cos_similar: CosSimilar

    def execute(self, text_1, text_2):
        wakati_text_1 = (" ").join(word_list_create([text_1], WORD_CLASS, STOP_WORDS)[0])
        wakati_text_2 = (" ").join(word_list_create([text_2], WORD_CLASS, STOP_WORDS)[0])
        docs = np.array([wakati_text_1, wakati_text_2])
        vector_1, vector_2 = self.tfidf_vectorize.execute(docs)
        result = self.cos_similar.execute(vector_1, vector_2)
        return result
