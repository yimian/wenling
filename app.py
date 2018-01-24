from flask import Flask, jsonify
from flask import abort
from flask import request
from flask_script import Manager

from src.predict_0 import *

# load the keras model
wen0_bleeding = load_model('wen0_multi.model')
tk_bleeding = get_tk('multi_pos.txt', 'multi_neg.txt', 'multi_neu.txt')
# wen0_alopecia = load_model('wen0_alopecia.model')
# tk_alopecia = get_tk('diaofa_pos.txt', 'diaofa_neu.txt', 'diaofa_neg.txt')


app = Flask(__name__)
manager = Manager(app)

@app.route('/')
def index():
    return 'Hello,world!'


@app.route('/todo/api/v1.0/review_predict', methods=['POST'])
def review_predict():
    if not request.json:
        abort(404)
    data = request.json
    text_list = data['text']
    predict_list = predict(wen0_bleeding, tk_bleeding, text_list)
    prediction = [{'predict': predict_list[i].tolist(), 'text': text_list[i]} for i in range(len(text_list))]
    data_predict = {
        'category': data['category'],
        'prediction': prediction
    }
    return jsonify(data_predict), 201


# @app.route('/todo/api/v1.0/reviews_predict', methods=['POST'])
# def reviews_predict():
#     if not request.json:
#         abort(404)
#     task = {
#         'category': request.json['category'],
#         'text': request.json['text'],
#     }
#     return jsonify(task), 201


if __name__ == '__main__':
    manager.run()
