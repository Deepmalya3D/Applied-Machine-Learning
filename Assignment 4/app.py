from flask import Flask, request
import joblib
from score import score
import json

model = joblib.load("./best_model.joblib")

app = Flask(__name__)

@app.route('/score', methods=['POST'])
def score_endpoint():
    data = json.loads(request.data)
    text = data['Text']
    pred, prop = score(text, model)
    return {'Prediction': int(pred), 'Propensity': float(prop)}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)