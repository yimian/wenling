# -*- coding: utf-8 -*-
from src import utils


class Params(object):
    """
    Parameters description:
        # Data Parameters
        pos_file_path: The path of positive sentence file.
        neu_file_path: The path of neural sentence file.
        neg_file_path: The path of negative sentence file.
        w2v_model_path: The path of word2vec model.

        # Random Seed
        random_seed: The seed to make sure that the data is shuffle in same way.

        # Input Parameters
        max_features: The size of the vocabulary.
        max_len: The max number of words in a sentences that the model use.
        embedding_size: The embedding size of word.
        dropout: Float between 0 and 1, which was used to reduce overfitting.


        # RNN Parameters
        output_size: Dimensionality of the output space.
        rnn_activation: Activation function to use.
        recurrent_activation: Activation function to use for the recurrent step.
        l2_regularization: Float, L2 regularization factor, which was used to reduce overfitting.

        # CNN Parameters
        num_filter: Integer, the dimensionality of the output space.
        filter_length: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
        padding: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
        cnn_activation: Activation function to use of CNN.
        pool_length: Integer, size of the max pooling windows.

        # Compile Parameters
        loss: str (name of objective function) or objective function.
        optimizer: str (name of optimizer) or optimizer object.

        # Training Parameters
        model_type: str. Name of model that is used, such as LSTM, BiLSTM, GRU, BiGRU, CNNLSTM
        batch_size: integer. Number of samples per gradient update.
        num_epoch: integer, the number of epochs to train the model.
        validation_split: float (0. < x < 1). Fraction of the data to use as held-out validation data.
        shuffle: boolean. Whether to shuffle the samples at each epoch.
        model_path: The path of model that we train.
        token_path: The path of text token which is used to transform the text to one-hot representation.

    """

    def __init__(self):
        # Data Parameters
        self.pos_file_path = ''
        self.neu_file_path = ''
        self.neg_file_path = ''
        self.w2v_model_path = ''

        # Random Seed
        self.random_seed = 0

        # Input Parameters
        self.max_features = 0
        self.max_len = 0
        self.embedding_size = 0
        self.dropout = 0

        # RNN Parameters
        self.output_size = 0
        self.rnn_activation = ''
        self.recurrent_activation = ''
        self.l2_regularization = 0

        # CNN Parameters
        self.filter_length = 0
        self.num_filter = 0
        self.padding = ''
        self.cnn_activation = ''
        self.pool_length = 0

        # Compile Parameters
        self.loss = ''
        self.optimizer = ''

        # Training Parameters
        self.model_type = ''
        self.batch_size = 0
        self.num_epoch = 0
        self.validation_split = 0
        self.shuffle = ''
        self.model_path = ''
        self.token_path = ''

        # Predict Parameters
        self.models_config = None

    def demo_init(self):
        # Data Parameters
        self.pos_file_path = utils.get_corpus_path('diaofa_pos.txt')
        self.neu_file_path = utils.get_corpus_path('diaofa_neu.txt')
        self.neg_file_path = utils.get_corpus_path('diaofa_neg.txt')
        self.w2v_model_path = utils.get_w2v_model_path('tm/tm.model')

        # Random Seed
        self.random_seed = 7

        # Input Parameters
        self.max_features = 5000
        self.max_len = 200
        self.embedding_size = 400
        self.dropout = 0.5
        self.l2_regularization = 0.05

        # RNN Parameters
        self.output_size = 50
        self.rnn_activation = 'tanh'
        self.recurrent_activation = 'hard_sigmoid'
        self.l2_regularization = 0

        # CNN Parameters
        self.filter_length = 3
        self.num_filter = 150
        self.padding = 'same'
        self.cnn_activation = 'relu'
        self.pool_length = 2

        # Compile Parameters
        self.loss = 'categorical_crossentropy'
        self.optimizer = 'sgd'

        # Training Parameters
        self.model_type = 'LSTM'
        self.batch_size = 256
        self.num_epoch = 1000
        self.validation_split = 0.2
        self.shuffle = True
        self.model_path = utils.get_model_path('diaofa.model') + '_' + self.model_type
        self.token_path = utils.get_model_path('diaofa.tk')

        # Predict Model Config
        self.models_config = {
            'mix': {
                'token_path' : utils.get_model_path('mix.tk'),
                'model_path': utils.get_model_path('mix.model_LSTM')
            },
            'diaofa': {
                'token_path': utils.get_model_path('diaofa.tk'),
                'model_path': utils.get_model_path('diaofa.model_LSTM')
            }
        }


params_o = Params()
params_o.demo_init()
