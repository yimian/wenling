import numpy
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import regularizers
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
from keras.preprocessing import text, sequence
import time
import gensim
import jieba
import numpy as np
import utils
import random

# Input parameters
max_features = 5000
max_len = 200
embedding_size = 300
border_mode = 'same'
dropout = 0.8
l2_regularization = 0.05

# RNN parameters
output_size = 50
rnn_activation = 'tanh'
recurrent_activation = 'hard_sigmoid'

# Compile parameters
loss = 'categorical_crossentropy'
optimizer = 'rmsprop'

# Training parameters
batch_size = 256
num_epoch = 100
validation_split = 0.2
shuffle = True

# random seed
r = 7

def training_save(model_name, pos_f_name, neg_f_name, neu_f_name):
    # Prepare the dataset
    l1 = []
    l2 = []
    pos_lines = open(utils.get_corpus_path(pos_f_name)).readlines()
    neg_lines = open(utils.get_corpus_path(neg_f_name)).readlines()
    neu_lines = open(utils.get_corpus_path(neu_f_name)).readlines()

    print('pos_num:', len(pos_lines))
    print('neg_num:', len(neg_lines))
    print('neu_num:', len(neu_lines))

    pos_cut_lines = list(map(lambda x: ' '.join(jieba.cut(x.replace('\n', ''))), pos_lines))
    neg_cut_lines = list(map(lambda x: ' '.join(jieba.cut(x.replace('\n', ''))), neg_lines))
    neu_cut_lines = list(map(lambda x: ' '.join(jieba.cut(x.replace('\n', ''))), neu_lines))
    # append pos and neg and neu
    for i in range(len(pos_cut_lines)):
        if i < len(neg_lines):
            l1.append(pos_cut_lines[i])
            l2.append([1, 0, 0])
            l1.append(neg_cut_lines[i])
            l2.append([0, 0, 1])
        else:
            l1.append(pos_cut_lines[i])
            l2.append([1, 0, 0])

    for i in range(len(neu_cut_lines)):
        l1.append(neu_cut_lines[i])
        l2.append([0, 1, 0])

    random.Random(7).shuffle(l1)
    random.Random(7).shuffle(l2)
    # random.shuffle(l1, lambda: r)
    # random.shuffle(l2, lambda: r)

    x = np.array(l1)
    y = np.array(l2)

    # Build vocabulary & sequences
    tk = text.Tokenizer(num_words=max_features, split=" ")
    tk.fit_on_texts(x)
    x = tk.texts_to_sequences(x)
    word_index = tk.word_index
    x = sequence.pad_sequences(x, maxlen=max_len)

    # Build pre-trained embedding layer
    w2v = gensim.models.Word2Vec.load(utils.get_w2v_model_path('dianping/dianping.model'))
    embedding_matrix = numpy.zeros((len(word_index) + 1, embedding_size))
    for word, i in word_index.items():
        if word in w2v.wv.vocab:
            embedding_matrix[i] = w2v[word]
    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_size,
                                weights=[embedding_matrix],
                                input_length=max_len)
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(dropout))
    model.add(LSTM(output_dim=output_size, activation=rnn_activation, recurrent_activation=recurrent_activation))
    model.add(Dropout(dropout))
    model.add(Dense(3, kernel_regularizer=regularizers.l2(l2_regularization)))
    model.add(Activation('softmax'))
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    print('============ LSTM w2v model training begin ===============')
    model.fit(x, y, batch_size=batch_size, epochs=num_epoch, validation_split=validation_split, shuffle=shuffle)
    print('============ LSTM w2v model training finish ==============')
    model.save(filepath=utils.get_model_path(model_name))
    print('model was saved to ' + utils.get_model_path(model_name))


if __name__ == '__main__':
    start_time = time.time()
    training_save('wen0_food.model', 'food_pos.txt', 'food_neg.txt', 'food_neu.txt')
    stop_time = time.time()
    print('Time used:', str(stop_time - start_time))
    # validation_split: 0.1, training set: 29436, test set: 3271, acc: 0.93, val_acc: 0.91  |  dropout: 0.8, l2_regularization: 0.05
    # validation_split: 0.2, training set: 26165, test set: 6542, acc: 0.93, val_acc: 0.90  |  dropout: 0.8, l2_regularization: 0.05
    # validation_split: 0.3, training set: 22894, test set: 9813, acc: 0.93, val_acc: 0.90  |  dropout: 0.8, l2_regularization: 0.05
    # validation_split: 0.4, training set: 19624, test set: 13083, acc: 0.94, val_acc: 0.90  |  dropout: 0.8, l2_regularization: 0.05
    # validation_split: 0.5, training set: 16353, test set: 16354, acc: 0.93, val_acc: 0.89  |  dropout: 0.8, l2_regularization: 0.05


