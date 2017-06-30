# WENLING Sentiment Analysis Model
A deep learning sentiment alaysis model base on word2vec and lstm.

## Config
The `config.py` contains the path where data and model was saved. And `utils.py` provides some methods to build the path.

## Data Processing
The `process_data.py` provides some methods to process the corpus such how to extract text in a `.xlsx` file and split it into positive, neural and negative file.

## Train Model
The `train_0.py` is the script to train the model. The parameters is set at the begin of the code. The model will be automatically saved to the path in `config.py`. All the packages you need is in `requirements.txt`. What's more, if your computer has gpu, you should install the `tensorflow-gpu` rather than `tensorflow`. 

## Predict
In `predict_0.py`, I show how the model can be used. Attention, your should use the `list` of text as input, and you will get a predict `list`. 