# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
# Add the necessary imports for the starter code.
import pandas as pd
import pickle
from ml.model import train_model, inference, compute_model_metrics
from sklearn.metrics import accuracy_score

# Add code to load in the data.
data = pd.read_csv("/home/omahfoudhi/PycharmProjects/pythonProject2/udacity_project3/starter/data/census.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
data.columns = data.columns.str.replace(' ', '')
train, test = train_test_split(data, test_size=0.20)

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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label='salary', training=True
)

"""preprocessor = ColumnTransformer(
    transformers=[
        ("cat", encoder, cat_features),
        ("label", lb, [" salary"]),
    ],
    remainder="drop",
)"""

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
# Train and save a model.
model = train_model(X_train, y_train)

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

with open('encoder.pkl', 'wb') as file:
    pickle.dump(encoder, file)

with open('lb.pkl', 'wb') as file:
    pickle.dump(lb, file)
y_pred = inference(model, X_test)

# Evaluate the model
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
print(f"precision: {precision}")
print(f"recall: {recall}")
print(f"fbeta: {fbeta}")