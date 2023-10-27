""" This module tests the root and the prediction end points """
from fastapi.testclient import TestClient
import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
# Import our app from main.py.
from main import app
from starter.ml.data import process_data
# Instantiate the testing client with our app.
client = TestClient(app)

DATA_PATH = 'data/clean_census.csv'
MODEL_PATH = 'starter/model.pkl'

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

@pytest.fixture(name='data')
def data():
    """
    Fixture will be used by the unit tests.
    """
    yield pd.read_csv(DATA_PATH)

def test_get_root():
    """ Test the root page get a succesful response"""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "Hello World": "Welcome in the third udacity project! looking forward to next projects"}


def test_post_predict_up():
    """ Test an example when income is less than 50K """

    r = client.post("/predict-income", json={
        "age": 37,
        "workclass": "Private",
        "fnlgt": 280464,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 80,
        "native_country": "United-States"
    })

    assert r.status_code == 200
    assert r.json() == {"Income prediction": ">50K"}


def test_post_predict_down():
    """ Test an example when income is higher than 50K """
    r = client.post("/predict-income", json={
        "age": 28,
        "workclass": "Private",
        "fnlgt": 183175,
        "education": "Some-college",
        "education_num": 10,
        "marital_status": "Divorced",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })

    assert r.status_code == 200
    assert r.json() == {"Income prediction": "<=50K"}


def test_load_data(data):
    """ Check the data received """

    assert isinstance(data, pd.DataFrame)
    assert data.shape[0] > 0
    assert data.shape[1] > 0


def test_model():
    """ Check model type """

    model = joblib.load(MODEL_PATH)
    assert isinstance(model, RandomForestClassifier)


def test_process_data(data):
    """ Test the data split """

    train, _ = train_test_split(data, test_size=0.20)
    X, y, _, _ = process_data(train, cat_features, label='salary')
    assert len(X) == len(y)