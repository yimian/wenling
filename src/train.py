# -*- coding: utf-8 -*-
"""

"""
import random
import time
import pickle
import jieba
import gensim
import numpy as np
from keras import regularizers

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
from keras.layers import GRU
from keras.layers import Conv1D, MaxPooling1D, Merge
from keras.preprocessing import text, sequence

from src.params import params_o


def base_for_train(params):
    x = []
    y = []
    with open(params.pos_file_path) as f:
        for line in f.readlines():
            x.append(' '.join(jieba.cut(line.replace('\n', ''))))
            y.append([1, 0, 0])
    with open(params.neu_file_path) as f:
        for line in f.readlines():
            x.append(' '.join(jieba.cut(line.replace('\n', ''))))
            y.append([0, 1, 0])
    with open(params.neg_file_path) as f:
        for line in f.readlines():
            x.append(' '.join(jieba.cut(line.replace('\n', ''))))
            y.append([0, 0, 1])

    random.Random(params.random_seed).shuffle(x)
    random.Random(params.random_seed).shuffle(y)

    x = np.array(x)
    y = np.array(y)

    tk = text.Tokenizer(num_words=params.max_features)
    tk.fit_on_texts(x)

    x = tk.texts_to_sequences(x)
    word_index = tk.word_index
    x = sequence.pad_sequences(x, maxlen=params.max_len)

    w2v = gensim.models.Word2Vec.load(params.w2v_model_path)
    embedding_matrix = np.zeros((len(word_index) + 1, params.embedding_size))
    for word, i in word_index.items():
        if word in w2v.wv.vocab:
            embedding_matrix[i] = w2v[word]
    embedding_layer = Embedding(len(word_index) + 1,
                                params.embedding_size,
                                weights=[embedding_matrix],
                                input_length=params.max_len)
    return x, y, tk, embedding_layer


def training(params):
    x, y, tk, embedding_layer = base_for_train(params)
    model = Sequential()
    model.add(embedding_layer)
    model_flag = True

    if params.model_type == 'LSTM':
        print('======== LSTM ========')
        model.add(Dropout(params.dropout))
        model.add(LSTM(units=params.output_size, activation=params.rnn_activation,
                       recurrent_activation=params.recurrent_activation))
        model.add(Dropout(params.dropout))
        model.add(Dense(3, kernel_regularizer=regularizers.l2(params_o.l2_regularization)))
        model.add(Activation('softmax'))
        model.compile(loss=params.loss,
                      optimizer=params.optimizer,
                      metrics=['accuracy'])
        model.fit(x, y, batch_size=params.batch_size, epochs=params.num_epoch, validation_split=params.validation_split,
                  shuffle=params.shuffle)

    elif params.model_type == 'GRU':
        print('======== GRU ========')
        model.add(GRU(units=params.output_size, activation=params.rnn_activation,
                      recurrent_activation=params.recurrent_activation))
        model.add(Dropout(params.dropout))
        model.add(Dense(3, kernel_regularizer=regularizers.l2(params_o.l2_regularization)))
        model.add(Activation('softmax'))

        model.compile(loss=params.loss,
                      optimizer=params.optimizer,
                      metrics=['accuracy'])

        model.fit(x, y, batch_size=params.batch_size, epochs=params.num_epoch,
                  validation_split=params.validation_split, shuffle=params.shuffle)

    elif params.model_type == 'BiLSTM':
        print('======== BiLSTM ========')
        model.add(Bidirectional(LSTM(units=params.output_size, activation=params.rnn_activation,
                                     recurrent_activation=params.recurrent_activation)))
        model.add(Dropout(params.dropout))
        model.add(Dense(3, kernel_regularizer=regularizers.l2(params_o.l2_regularization)))
        model.add(Activation('softmax'))

        model.compile(loss=params.loss,
                      optimizer=params.optimizer,
                      metrics=['accuracy'])

        model.fit(x, y, batch_size=params.batch_size, epochs=params.num_epoch,
                  validation_split=params.validation_split, shuffle=params.shuffle)

    elif params.model_type == 'BiGRU':
        print('======== BiGRU ========')
        model.add(Bidirectional(GRU(units=params.output_size, activation=params.rnn_activation,
                                    recurrent_activation=params.recurrent_activation)))
        model.add(Dropout(params.dropout))
        model.add(Dense(3, kernel_regularizer=regularizers.l2(params_o.l2_regularization)))
        model.add(Activation('softmax'))

        model.compile(loss=params.loss,
                      optimizer=params.optimizer,
                      metrics=['accuracy'])

        model.fit(x, y, batch_size=params.batch_size, epochs=params.num_epoch,
                  validation_split=params.validation_split, shuffle=params.shuffle)

    elif params.model_type == 'CNNLSTM':
        print('======== CNNLSTM ========')
        model.add(Dropout(params.dropout))
        model.add(Conv1D(filters=params.num_filter,
                         kernel_size=params.filter_length,
                         padding=params.border_mode,
                         activation=params.cnn_activation))
        model.add(MaxPooling1D(pool_size=params.pool_length))
        model.add(Bidirectional(LSTM(units=params.output_size, activation=params.rnn_activation,
                                     recurrent_activation=params.recurrent_activation)))
        model.add(Dropout(params.dropout))
        model.add(Dense(3, kernel_regularizer=regularizers.l2(params_o.l2_regularization)))
        model.add(Activation('softmax'))

        model.compile(loss=params.loss,
                      optimizer=params.optimizer,
                      metrics=['accuracy'])

        model.fit(x, y, batch_size=params.batch_size, epochs=params.num_epoch,
                  validation_split=params.validation_split, shuffle=params.shuffle)

    else:
        print('======== There is not this model ========')
        model_flag = False

    if model_flag:
        model.save(filepath=params.model_path)
        pickle.dump(tk, open(params.token_path, 'wb'))
        print('model was saved to ' + params.model_path)


if __name__ == '__main__':
    start_time = time.time()
    training(params_o)
    stop_time = time.time()
    print('Time used:', str(stop_time - start_time))
