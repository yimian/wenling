# -*- coding: utf-8 -*-
"""

"""
import random
import time
import pickle
import jieba
import gensim
import numpy as np
# from keras import regularizers

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.layers import Convolution1D, MaxPooling1D, Merge
from keras.preprocessing import text, sequence

from src.param import params_o


# # Input parameters
# max_features = 5000
# max_len = 200
# embedding_size = 400
# border_mode = 'same'
# dropout = 0.25
# l2_regularization = 0.05
#
# # RNN parameters
# output_size = 50
# rnn_activation = 'tanh'
# recurrent_activation = 'hard_sigmoid'
#
# # Compile parameters
# loss = 'categorical_crossentropy'
# optimizer = 'rmsprop'
#
# # Training parameters
# batch_size = 512
# num_epoch = 10
# validation_split = 0.2
# shuffle = True
#
# # random seed
# r = 7


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
    # Prepare the dataset
    # l1 = []
    # l2 = []
    #
    # with open(utils.get_corpus_path(pos_f_name)) as f:
    #     for line in f.readlines():
    #         l1.append(' '.join(jieba.cut(line.replace('\n', ''))))
    #         l2.append([1, 0, 0])
    #
    # with open(utils.get_corpus_path(neu_f_name)) as f:
    #     for line in f.readlines():
    #         l1.append(' '.join(jieba.cut(line.replace('\n', ''))))
    #         l2.append([0, 1, 0])
    #
    # with open(utils.get_corpus_path(neg_f_name)) as f:
    #     for line in f.readlines():
    #         l1.append(' '.join(jieba.cut(line.replace('\n', ''))))
    #         l2.append([0, 0, 1])
    #
    # random.Random(r).shuffle(l1)
    # random.Random(r).shuffle(l2)
    #
    # x = np.array(l1)
    # y = np.array(l2)

    # x, y = read_data(params)
    #
    # # Build vocabulary & sequences
    # tk = text.Tokenizer(num_words=max_features)
    # tk.fit_on_texts(x)
    # x = tk.texts_to_sequences(x)
    # word_index = tk.word_index
    # x = sequence.pad_sequences(x, maxlen=max_len)
    #
    # # Build pre-trained embedding layer
    # w2v = gensim.models.Word2Vec.load(utils.get_w2v_model_path(w2v_model_name))
    # embedding_matrix = np.zeros((len(word_index) + 1, embedding_size))
    # for word, i in word_index.items():
    #     if word in w2v.wv.vocab:
    #         embedding_matrix[i] = w2v[word]
    # embedding_layer = Embedding(len(word_index) + 1,
    #                             embedding_size,
    #                             weights=[embedding_matrix],
    #                             input_length=max_len)

    x, y, tk, embedding_layer = base_for_train(params)
    model = Sequential()
    model.add(embedding_layer)

    if params.model_type == 'LSTM':
        model.add(Dropout(params.dropout))
        model.add(LSTM(units=params.output_size, activation=params.rnn_activation,
                       recurrent_activation=params.recurrent_activation))
        model.add(Dropout(params.dropout))
        # model.add(Dense(3, kernel_regularizer=regularizers.l2(l2_regularization)))
        model.add(Dense(3))
        model.add(Activation('softmax'))
        model.compile(loss=params.loss,
                      optimizer=params.optimizer,
                      metrics=['accuracy'])
        model.fit(x, y, batch_size=params.batch_size, epochs=params.num_epoch, validation_split=params.validation_split,
                  shuffle=params.shuffle)

    if params.model_type == 'GRU':
        model.add(GRU(output_dim=params.output_size, activation=params.rnn_activation,
                      recurrent_activation=params.recurrent_activation))
        model.add(Dropout(params.dropout))
        model.add((Dense(3)))
        model.add(Activation('softmax'))

        model.compile(loss=params.loss,
                      optimizer=params.optimizer,
                      metrics=['accuracy'])

        model.fit(x, y, batch_size=params.batch_size, nb_epoch=params.num_epoch,
                  validation_split=params.validation_split, shuffle=params.shuffle)

    if params.model_type == 'BiLSTM':
        model.add(Bidirectional(LSTM(output_dim=params.output_size, activation=params.rnn_activation,
                                     recurrent_activation=params.recurrent_activation)))
        model.add(Dropout(params.dropout))
        model.add((Dense(3)))
        model.add(Activation('softmax'))

        model.compile(loss=params.loss,
                      optimizer=params.optimizer,
                      metrics=['accuracy'])

        model.fit(x, y, batch_size=params.batch_size, nb_epoch=params.num_epoch,
                  validation_split=params.validation_split, shuffle=params.shuffle)

    if params.model_type == 'BiGRU':
        model.add(Bidirectional(GRU(output_dim=params.output_size, activation=params.rnn_activation,
                                    recurrent_activation=params.recurrent_activation)))
        model.add(Dropout(params.dropout))
        model.add((Dense(3)))
        model.add(Activation('softmax'))

        model.compile(loss=params.loss,
                      optimizer=params.optimizer,
                      metrics=['accuracy'])

        model.fit(x, y, batch_size=params.batch_size, nb_epoch=params.num_epoch,
                  validation_split=params.validation_split, shuffle=params.shuffle)

    if params.model_type == 'CNNLSTM':
        model.add(Dropout(params.dropout))
        model.add(Convolution1D(nb_filter=params.num_filter,
                                filter_length=params.filter_length,
                                border_mode=params.border_mode,
                                activation=params.cnn_activation,
                                subsample_length=1))
        model.add(MaxPooling1D(pool_length=params.pool_length))
        model.add(Bidirectional(LSTM(output_dim=params.output_size, activation=params.rnn_activation,
                                     recurrent_activation=params.recurrent_activation)))
        model.add(Dropout(params.dropout))
        model.add((Dense(3)))
        model.add(Activation('softmax'))

        model.compile(loss=params.loss,
                      optimizer=params.optimizer,
                      metrics=['accuracy'])

        model.fit(x, y, batch_size=params.batch_size, nb_epoch=params.num_epoch,
                  validation_split=params.validation_split, shuffle=params.shuffle)

    model.save(filepath=params.model_path)
    pickle.dump(tk, open(params.token_path, 'wb'))
    print('model was saved to ' + params.model_path)


if __name__ == '__main__':
    start_time = time.time()
    training(params_o)
    stop_time = time.time()
    print('Time used:', str(stop_time - start_time))
