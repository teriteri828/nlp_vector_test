import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from util_m import CosSimilar
from lda_vectorize import LdaVectorize
from dataclasses import dataclass
import gensim


@dataclass
class LdaCosCalc:
    lda_vectorize: LdaVectorize
    cos_similar: CosSimilar

    def execute(self, text_1, text_2):
        vector_1 = self.lda_vectorize.execute(text_1)
        vector_2 = self.lda_vectorize.execute(text_2)
        result = self.cos_similar.execute(vector_1, vector_2)
        return result
