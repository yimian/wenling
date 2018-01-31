# -*- coding: utf-8 -*-
import csv
import re
import jieba
import pickle
import keras
from keras.preprocessing import sequence
from src.params import params_o


class Predictor(object):
    def __init__(self):
        self.tk = None
        self.keras_model = None
        self.name = None
        self.max_len = params_o.max_len

    def predict(self, sentence_list):
        x = [' '.join((jieba.cut(sentence))) for sentence in sentence_list]
        x = self.tk.texts_to_sequences(x)
        x = sequence.pad_sequences(x, maxlen=self.max_len)
        result = self.keras_model.predict(x).tolist()
        return result


def init_predictor_dict():
    predictor_dict = {}
    for name in params_o.models_config:
        predictor = Predictor()
        predictor.tk = pickle.load(open(params_o.models_config[name]['token_path'], 'rb'))
        predictor.keras_model = keras.models.load_model(filepath=params_o.models_config[name]['model_path'])
        predictor_dict[name] = predictor
    return predictor_dict


def index_to_label(index):
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
    predictor_dict = init_predictor_dict()
    for name in predictor_dict:
        print(name)
        predictor = predictor_dict[name]
        y_predict = predictor.predict(x)

        y_predict_label = list(
            map(lambda x: label_list[x.index(max(x))], y_predict))

        error_count = 0
        for i in range(len(y_label)):
            if y_predict_label[i] != y_label[i]:
                error_count += 1
        print('Accuracy:')
        print((len(y_label) - error_count) / len(y_label))
