from flask import Flask, jsonify
from flask import abort
from flask import request
from flask_script import Manager
from src.predict import init_predictor_dict

predictor_dict = init_predictor_dict()
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
    predict_list = predictor_dict[data['category']].predict(text_list)
    prediction = [{'predict': predict_list[i], 'text': text_list[i]}
                  for i in range(len(text_list))]
    data_predict = {
        'category': data['category'],
        'prediction': prediction
    }
    return jsonify(data_predict)


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
