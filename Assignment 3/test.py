import joblib
from score import score
import numpy as np
import os
import requests
import unittest
import time


class TestScore(unittest.TestCase):
    def setUp(self):
        self.model = joblib.load("./best_model.joblib")
        self.text = "Click here and win a holiday trip to Malibu. Limited time offer."
        self.thresh = 0.5
        self.pred, self.prop = score(self.text, self.model, self.thresh)

    # check if score function produces some output without crashing
    def test_smoke(self):
        assert self.pred != None
        assert self.prop != None
            
    # check if the input/output formats are as expected
    def test_format(self):
        assert type(self.text) == str
        assert type(self.thresh) == float 
        assert type(self.pred) == type(1)
        assert type(self.prop) == np.ndarray

    # check if the prediction value lies in {0,1}
    def test_pred_value(self):
        assert self.pred == 0 or self.pred == 1

    # check if propensity score lies in [0,1]
    def test_propensity_value(self):
        assert self.prop >= 0 and self.prop <= 1

    # if threshold is 0, prediction becomes 1
    def test_pred_test_0(self):
        pred, prop = score(self.text, self.model, 0)
        assert pred == 1

    # if threshold is 1, prediction becomes 0
    def test_pred_test_1(self):
        pred, prop = score(self.text, self.model, 1)
        assert pred == 0

    # testing obvious spam
    def test_spam(self):
        pred, prop = score("Click here to claim your prize, one million dollars.", self.model)
        assert pred == 1

    # testing obvious non-spam
    def test_nspam(self):
        pred, prop = score("Hi, how are you?", self.model)
        assert pred == 0


class TestFlaskIntegration(unittest.TestCase):
    def setUp(self):
        self.text = "Click here and win a holiday trip to Malibu. Limited time offer."
        os.system('python app.py &')
        time.sleep(1)

    def tearDown(self):
        os.system("kill $(lsof -t -i:5000)")

    def test_flask(self):
        payload = {'Text': self.text}
        response = requests.post('http://127.0.0.1:5000/score', json = payload)
        data = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn('Prediction', data)
        self.assertIn('Propensity', data)

if __name__ == '__main__':
    unittest.main()