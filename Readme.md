# From DS to MLOPs

From Data Science to MLOPs workshop

# Dataset
## Boston Housing (Predict Prices) Data Set

For this workshop we are going to work with the following dataset:

https://kaggle.com/c/house-prices-advanced-regression-techniques/overview (Predict Prices)

Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence. \

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

### Skills Seveloped:

1) EDA
2) Feature Engineering
3) Modeling
4) Pipelines
5) Deployment with Flask

# Virtual Environment

Firt we need to create a virtual environment for the project, to keep track of every dependency, it is also useful to use and explicit version of Python

Install the package for creating a virtual environment:

`$ pip install virtualenv`

Create a new virtual environment

`$ virtualenv venv`

Activate virtual environment

`$ source venv/bin/activate`

# Python packages

Now with the virtual environment we can install the dependencies written in requirements.txt

`$ pip install -r requirements.txt`

# Train

After we have install all the dependencies we can now run the script in code/train.py, this script takes the input data and outputs a trained model and a pipeline for our web service.

`$ python code/train.py`

# Web application

Finally we can test our web application by running:

`$ python app.py`

# Test!

Test by using the calls in tests/example_calls.txt from the terminal