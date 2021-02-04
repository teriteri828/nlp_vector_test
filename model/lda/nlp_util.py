import MeCab

# mecabのインスタンス生成
mecab = MeCab.Tagger("-Ochasen")
mecab.parse("")


def word_list_create(sentences, get_word_class, stop_words):
    # 複数文から、指定の品詞(GET_WORD_CLASS)を抽出した単語リスト
    sentences_word_list = []

    for sentence in sentences:
        # 一文から、指定の品詞(GET_WORD_CLASS)を抽出した単語リスト
        one_sentence_word_list = []
        # 形態素解析
        node = mecab.parseToNode(sentence)

        while node:

            # 語幹
            word = node.feature.split(",")[-3]
            # 品詞
            word_class = node.feature.split(",")[0]
            # (指定の品詞(GET_WORD_CLASS)である) and (語幹が＊のもの(つまり未知語))場合は、単語リストに追加
            if word_class in get_word_class and word != "*" and not word in stop_words:
                one_sentence_word_list.append(word)

            node = node.next
        sentences_word_list.append(one_sentence_word_list)
    return sentences_word_list
