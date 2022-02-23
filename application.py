# Local imports
import datetime
from urllib import response

# Third part imports
from flask import request

from microservices import application
from microservices.functions import get_model_response

model_name = 'House Price Prediction'
model_file = 'model_binary.dat.gz'
version = 'v1.0.0'


@application.route('/info', methods=['GET'])
def info():
    """ Return model information, version, how to call """
    result = {'name': model_name, 'version': version}

    return result


@application.route('/health', methods=['GET'])
def health():
    """Return service health"""
    return 'ok'


@application.route('/predict', methods=['POST'])
def predict():
    feature_dict = request.get_json()
    if not feature_dict:
        return {
            'error': 'Body is empty.'
            }, 500

    try:
        response_ = get_model_response(feature_dict)
    except ValueError as e:
        return {'error': str(e).split('\n')[-1].strip()}, 500

    return response_, 200


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000, debug=False)
