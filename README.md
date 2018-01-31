# WENLING (文灵) Sentiment Analysis Package
A deep learning sentiment alaysis package which includes several models such as LSTM, Bi-LSTM, GRU, Bi-GRU and CNN-LSTM.

## Config & Utils
The `config.py` contains the path where data and model are saved.

The `utils.py` provides some methods to build the path.

## Data Processing
The `process_data.py` provides some methods to process the corpus such as how to extract text from a `.xlsx` file and split it into positive, neural and negative file.

## Parameters
The `params.py` defines all the parameters which are needed in training and predicting.

To make sure the predicting and training use the same parameters, I new a parameters object `params_o` in `params.py`
and initialize its parameters. When training and predicting, you should import the same parameters object. You should
change the parameters in `params.py` instead of in training or predicting.

## Install
```angular2html
$ pip install -r requirements.txt
```

## Configure the Parameters
Configure the corpus path, model path and other parameters and predicting in training in `params.py`.

## Train Model
Just run the command in root directory of project:
```
$ python -m src.train
```

## Predict
```angular2html
$ python -m src.predict
```

## Models Measure
training set: 26165 samples

batch size: 256 samples

Time (each iteration):
- LSTM: 36s 
- Bi-LSTM: 67s 
- GRU: 27s 
- Bi-GRU: 52s
- CNN-LSTM: 39s

Accuracy: 90% (about) for all models after 10 iterations
