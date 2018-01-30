# WENLING Sentiment Analysis Package
A deep learning sentiment alaysis package which includes several models such as LSTM, Bi-LSTM, GRU, Bi-GRU and CNN-LSTM.

## Config & Utils
The `config.py` contains the path where data and model are saved.

The `utils.py` provides some methods to build the path.

## Data Processing
The `process_data.py` provides some methods to process the corpus such as how to extract text from a `.xlsx` file and split it into positive, neural and negative file.

## Parameters
The `params.py` defines all the parameters which are needed in training and predicting.

## Train Model
Just run:
```
python train.py
```

## Predict



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
