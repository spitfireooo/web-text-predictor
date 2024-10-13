from django.shortcuts import render
from django.views.decorators.http import require_http_methods


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from tensorflow.keras.layers import Dense, SimpleRNN, Input, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.utils import to_categorical

def text_predictor(text):
    print("PATH: ", os.getcwd())
    with open('text.txt', 'r', encoding='utf-8') as f:
        texts = f.read()
        texts = texts.replace('\ufeff', '')

    maxWordsCount = 1000
    tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                               lower=True, split=' ', char_level=False)
    tokenizer.fit_on_texts([texts])

    dist = list(tokenizer.word_counts.items())
    print(dist[:10])

    data = tokenizer.texts_to_sequences([texts])
    res = np.array(data[0])

    inp_words = 3
    n = res.shape[0] - inp_words

    X = np.array([res[i:i + inp_words] for i in range(n)])
    Y = to_categorical(res[inp_words:], num_classes=maxWordsCount)

    model = Sequential()
    model.add(Embedding(maxWordsCount, 256, input_length=inp_words))
    model.add(SimpleRNN(128, activation='tanh'))
    model.add(Dense(maxWordsCount, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    history = model.fit(X, Y, batch_size=32, epochs=50)

    def word_build(texts, str_len=20):
        res = texts
        data = tokenizer.texts_to_sequences([texts])[0]
        for i in range(str_len):
            x = data[i: i + inp_words]
            inp = np.expand_dims(x, axis=0)

            pred = model.predict(inp)
            idx = pred.argmax(axis=1)[0]
            data.append(idx)

            res += " " + tokenizer.index_word[idx]
        return res

    return word_build(text)

@require_http_methods(['POST', 'GET'])
def index(request):
    if request.method == 'GET':
        return render(request, 'webtextpredictor/index.html', {'title': 'Web Text Predictor'})

    elif request.method == 'POST':
        text = request.POST['text']
        if text:
            result = text_predictor(text)
            return render(request, 'webtextpredictor/index.html',
                          {'title': 'Web Text Predictor', 'text': text, 'result': result})
        else:
            return render(request, 'webtextpredictor/index.html',
                          {'title': 'Web Text Predictor', 'text': text, 'result': 'Вы ничего не ввели, повторите попытку...'})
