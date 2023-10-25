from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

cat_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country'
]
def train_model(path):
    data = pd.read_csv(path)

    X_train, y_train, X_test, y_test = train_test_split(
        data.drop(columns=['salary']), data['salary'],
        test_size=0.2,
        stratify=data['salary'],
    )
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(y_train.values.reshape(-1, 1))
    y_train = ordinal_encoder.transform(y_train.values.reshape(-1, 1))


    preprocess = ColumnTransformer(
        transformers=[
            ('ordinal', ordinal_encoder, cat_columns)
        ],
        remainder='drop'
    )

    sk_pipe = Pipeline(
        steps=[
            ('preprocess', preprocess),
            ('classifier', RandomForestClassifier())
        ]
    )
    sk_pipe.fit(X_train, y_train)



if __name__ == '__main__':
    train_model('starter/data/census.csv')