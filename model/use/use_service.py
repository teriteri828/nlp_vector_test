import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from util_m import CosSimilar
from use_vectorize import UseVectorize
from dataclasses import dataclass
import tensorflow_hub as hub
import tensorflow_text


@dataclass
class UseCosCalc:
    use_vectorize: UseVectorize
    cos_similar: CosSimilar

    def execute(self, text_1, text_2):
        vector_1 = self.use_vectorize.execute(text_1)
        vector_2 = self.use_vectorize.execute(text_2)
        result = self.cos_similar.execute(vector_1, vector_2)
        return result
