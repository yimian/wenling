# -*- coding: utf-8 -*-
from src import utils


class Params(object):
    def __init__(self):
        # dataset parameters
        self.pos_file_path = ''
        self.neu_file_path = ''
        self.neg_file_path = ''
        self.w2v_model_path = ''
        self.model_path = ''
        self.token_path = ''
        # random seed
        self.random_seed = 0

        # Input parameters
        self.max_features = 0
        self.max_len = 0
        self.embedding_size = 0

        self.dropout = 0
        self.l2_regularization = 0

        # RNN parameters
        self.output_size = 0
        self.rnn_activation = ''
        self.recurrent_activation = ''

        # CNN parameters
        self.filter_length = 0
        self.num_filter = 0
        self.pool_length = 0
        self.cnn_activation = ''
        self.border_mode = ''

        # Compile parameters
        self.loss = ''
        self.optimizer = ''

        # Training parameters
        self.model_type = ''
        self.batch_size = 0
        self.num_epoch = 0
        self.validation_split = 0
        self.shuffle = ''

    def demo_init(self):
        # data parameters
        self.pos_file_path = utils.get_corpus_path('diaofa_pos.txt')
        self.neu_file_path = utils.get_corpus_path('diaofa_neu.txt')
        self.neg_file_path = utils.get_corpus_path('diaofa_neg.txt')
        self.w2v_model_path = utils.get_w2v_model_path('tm/tm.model')
        self.model_path = utils.get_model_path('wen0_diaofa.model') + '_' + self.model_type
        self.token_path = utils.get_model_path('wen0_diaofa.tk')

        # random seed
        self.random_seed = 7

        # Input parameters
        self.max_features = 5000
        self.max_len = 200
        self.embedding_size = 400
        self.dropout = 0.25
        self.l2_regularization = 0.05

        # RNN parameters
        self.output_size = 50
        self.rnn_activation = 'tanh'
        self.recurrent_activation = 'hard_sigmoid'

        # CNN parameters
        self.filter_length = 3
        self.num_filter = 150
        self.pool_length = 2
        self.cnn_activation = 'relu'
        self.border_mode = 'same'

        # Compile parameters
        self.loss = 'categorical_crossentropy'
        self.optimizer = 'rmsprop'

        # Training parameters
        self.model_type = 'LSTM'
        self.batch_size = 512
        self.num_epoch = 10
        self.validation_split = 0.2
        self.shuffle = True


params_o = Params()
params_o.demo_init()