# -*- coding: utf-8 -*-
import csv
import re
import jieba
import keras
import pickle
import numpy as np
from keras.preprocessing import text, sequence
from src import utils


class Predictor(object):

    def __init__(self):
        self.max_features = 5000
        self.max_len = 200
        self.embedding_size = 400
        self.keras_model_name = 'wen0_diaofa'

        # the pre-train keras model
        self.keras_model = self.load_model(self.keras_model_name + '.model')
        # get the pre-train tk
        self.tk = self.get_tk(self.keras_model_name + '.tk')

    def get_tk(self, tk_name):
        tk = pickle.load(open(utils.get_model_path(tk_name), 'rb'))
        return tk

    def load_model(self, model_name):
        print('Loading model.')
        model = keras.models.load_model(
            filepath=utils.get_model_path(model_name))
        print('Loading finish.')
        return model

    def input_transform(self, slist):
        l0 = []
        for ss in slist:
            string0 = ' '.join(jieba.cut(ss.replace('\n', '')))
            l0.append(string0)
        l0 = np.array(l0)
        input0 = self.tk.texts_to_sequences(l0)
        input1 = sequence.pad_sequences(input0, maxlen=self.max_len)
        return input1

    def predict(self, string_list):
        input1 = self.input_transform(string_list)
        result = self.keras_model.predict(input1).tolist()
        return result

    def index_to_label(self, index):
        if index == 0:
            return 5
        elif index == 1:
            return 3
        elif index == 2:
            return 1


if __name__ == '__main__':
    o_predictor = Predictor()

    label_list = ['5', '3', '1']
    x = []
    y_label = []

    with open('data/tuofa_comments.csv') as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            text = row[1]
            rate = re.findall(r'\d+\.?\d*', row[2])[0]
            x.append(text.replace('脱发', '掉发'))
            y_label.append(rate)

    y_predict = o_predictor.predict(x).tolist()

    for i in range(30):
        print(x[i], y_predict[i])

    y_predict_label = list(
        map(lambda x: label_list[x.index(max(x))], y_predict))

    error_count = 0
    for i in range(len(y_label)):
        if y_predict_label[i] != y_label[i]:
            error_count += 1

    print('Accuracy:')
    print((len(y_label) - error_count) / len(y_label))
