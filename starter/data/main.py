import pandas as pd
from sklearn.model_selection import train_test_split

def read_data(path):
    """
    Reads the data from the csv file and returns a pandas dataframe
    """
    data = pd.read_csv(path)
    return data

def clean_data(data):
    """
    Cleans the data and returns a pandas dataframe
    """
    data = data.dropna()
    return data

def split_data(data):
    """
    Splits the data into training and testing data
    """
    trainval, test = train_test_split(
        data,
        test_size=0.2,
        stratify='salary',
    )
    return trainval, test

if __name__ == '__main__':
    data = read_data('starter/data/census.csv')
    data = clean_data(data)
    print(data.info())