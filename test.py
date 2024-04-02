import score
import numpy
import os
import requests
import subprocess
import time
import unittest
import pickle
import pytest


@pytest.fixture

def model():    
    with open('logistic_regression.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


def test_smoke(model):
    threshold=0.5
    sent="Congratulations! You have won a free ticket to the cinema!"

    label,prop=score.score(sent,model,threshold)
    assert label in [0,1]
    assert 0 <= prop <=1


def test_format(model):
    threshold=0.5
    sent="Congratulations! You have won a free ticket to the cinema!"

    label,prop=score.score(sent,model,threshold)

    assert isinstance(label, int)
    assert isinstance(prop, float)
  
def test_prediction_value(model):
    text = "Sample text"
    threshold = 0.5
    prediction, _ = score.score(text, model, threshold)
    assert prediction in [True, False]

def test_propensity_score(model):
    text = "Sample text"
    threshold = 0.5
    _, propensity = score.score(text, model, threshold)
    assert propensity >= 0
    assert propensity <= 1

def test_threshold_0(model):
    threshold=0
    sent="Congratulations! You have won a free ticket to the cinema!"
    label,prop=score.score(sent,model,threshold)
    assert label==1

def test_threshold_1(model):
    threshold=1
    sent="Congratulations! You have won a free ticket to the cinema!"
    label,prop=score.score(sent,model,threshold)
    assert label==0

def test_spam(model):
    threshold=0.5
    sent="Congratulations! You have won a free ticket to the cinema!"
    label,prop=score.score("YOU HAVE WON 1 MILLION DOLLARS. SEND YOUR ACCOUNT DETAILS!",model,threshold)
    assert label == 1

def test_ham(model):
    threshold=1
    sent="Its a real mail, not spam"
    label,prop=score.score("Dogs are better than cats anyday.",model,threshold)
    assert label == 0


# integration test function
def test_flask():
    # Launch the Flask app using os.system
    os.system('start /b python app.py')

    # Wait for the app to start up
    time.sleep(15)

    # Make a request to the endpoint
    response = requests.get('http://127.0.0.1:5000/')
    print(response.status_code)
    
    # Checking if the Flask app is properly configured and running or not.
    assert response.status_code == 200
    assert type(response.text)== str

    # Shut down the Flask app using os.system
    os.system('kill $(lsof -t -i:5000)')