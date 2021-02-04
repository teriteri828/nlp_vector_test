from dataclasses import dataclass
import tensorflow


@dataclass
class UseVectorize:

    vectorizer: tensorflow

    def execute(self, text):

        result = self.vectorizer(text)[0]
        return result
