from flask import Flask
import joblib

# Iniciate app
app = Flask(__name__)

# Load models
model = joblib.load('models/model_binary.dat.gz')
