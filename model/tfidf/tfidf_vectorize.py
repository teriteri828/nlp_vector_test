from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass


@dataclass
class TfidfVectorize:
    vectorizer: TfidfVectorizer

    def execute(self, docs):
        tfidf_model = self.vectorizer(use_idf=True, token_pattern=u"(?u)\\b\\w+\\b")
        vecs = tfidf_model.fit_transform(docs)
        result = vecs.toarray()
        return result[0], result[1]
