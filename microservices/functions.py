import pandas as pd
from microservices import model


def predict(x, model_):
    prediction = model_.predict(x)
    return prediction


def get_model_response(json_data):
    x = pd.DataFrame.from_dict(json_data)
    prediction = predict(x, model)
    return {
        'status': 200,
        'prediction': prediction[0],
    }
