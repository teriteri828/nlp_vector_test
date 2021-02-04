import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "./model"))
sys.path.append(os.path.join(os.path.dirname(__file__), "./model/use"))
sys.path.append(os.path.join(os.path.dirname(__file__), "./model/tfidf"))
sys.path.append(os.path.join(os.path.dirname(__file__), "./model/lda"))


from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_bootstrap import Bootstrap
from dataclasses import dataclass
import tensorflow_hub as hub
import tensorflow_text
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim

from model.util_m import CosSimilar
from model.use.use_vectorize import UseVectorize
from model.use.use_service import UseCosCalc
from model.tfidf.tfidf_vectorize import TfidfVectorize
from model.tfidf.tfidf_service import TfidfCosCalc
from model.lda.lda_vectorize import LdaVectorize
from model.lda.lda_service import LdaCosCalc

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

app = Flask(__name__)
bootstrap = Bootstrap(app)


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/cos_calc", methods=["GET", "POST"])
def use_index():
    if request.form.get("text_1") == None and request.form.get("text_2") == None:
        return render_template("index.html")
    text_1 = request.form.get("text_1")
    text_2 = request.form.get("text_2")

    use_cos_calc = UseCosCalc(UseVectorize(embed), CosSimilar())
    use_cos = use_cos_calc.execute(text_1, text_2)

    tfidf_cos_calc = TfidfCosCalc(TfidfVectorize(TfidfVectorizer), CosSimilar())
    tfidf_cos = tfidf_cos_calc.execute(text_1, text_2)

    dictionary = gensim.corpora.Dictionary.load_from_text("./model/lda/data/text.dict")
    lda = gensim.models.LdaModel.load("./model/lda/data/lda.model")
    lda_cos_calc = LdaCosCalc(LdaVectorize(lda, dictionary), CosSimilar())
    lda_cos = lda_cos_calc.execute(text_1, text_2)

    return render_template(
        "index.html",
        use_cos=use_cos,
        tfidf_cos=tfidf_cos,
        lda_cos=lda_cos,
        text_1=text_1,
        text_2=text_2,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
