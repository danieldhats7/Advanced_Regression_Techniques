import pandas as pd
from microservices import model
from utils.GetData import DataPrep


def predict(x, model_):
    df = pd.read_csv('../in/train.csv')
    y = df['SalePrice']
    df.drop('SalePrice', axis=1, inplace=True)
    data = DataPrep()
    _, __ = data.get_data(df.copy(), y)
    x_, _ = data.get_data(x, fit=False)
    prediction = model_.predict(x_)[0]
    return prediction


def get_model_response(json_data):
    x = pd.DataFrame.from_dict(json_data)
    prediction = predict(x, model)
    return {
        'status': 200,
        'prediction': prediction,
    }
