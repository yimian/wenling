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
    predict_list = predictor_dict[data['model_name']].predict(text_list)
    prediction = [{'predict': predict_list[i], 'text': text_list[i]}
                  for i in range(len(text_list))]
    data_predict = {
        'model_name': data['model_name'],
        'prediction': prediction
    }
    return jsonify(data_predict)


if __name__ == '__main__':
    manager.run()
