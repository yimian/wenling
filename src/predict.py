# -*- coding: utf-8 -*-
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


