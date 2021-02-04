import gensim

lda = gensim.models.LdaModel.load("./data/lda.model")

for i in range(50):
    print("tpc_{0}: {1}".format(i, lda.print_topic(i)[0:80] + "..."))
