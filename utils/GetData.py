import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

def get_data(PATH):
    
    df = pd.read_csv(PATH)