import pandas as pd
from microservices import model
import joblib


def predict(x, model_):
    prediction = model_.predict(x)
    return prediction


def get_model_response(json_data):
    #model = joblib.load('models/kaggle.dat.gz')
    #data = [{"Id":1461,"MSSubClass":20,"MSZoning":"RH","LotFrontage":80.0,"LotArea":11622,"Street":"Pave","Alley":"Grvl","LotShape":"Reg","LandContour":"Lvl","Utilities":"AllPub","LotConfig":"Inside","LandSlope":"Gtl","Neighborhood":"NAmes","Condition1":"Feedr","Condition2":"Norm","BldgType":"1Fam","HouseStyle":"1Story","OverallQual":5,"OverallCond":6,"YearBuilt":1961,"YearRemodAdd":1961,"RoofStyle":"Gable","RoofMatl":"CompShg","Exterior1st":"VinylSd","Exterior2nd":"VinylSd","MasVnrType":"None","MasVnrArea":0.0,"ExterQual":"TA","ExterCond":"TA","Foundation":"CBlock","BsmtQual":"TA","BsmtCond":"TA","BsmtExposure":"No","BsmtFinType1":"Rec","BsmtFinSF1":468.0,"BsmtFinType2":"LwQ","BsmtFinSF2":144.0,"BsmtUnfSF":270.0,"TotalBsmtSF":882.0,"Heating":"GasA","HeatingQC":"TA","CentralAir":"Y","Electrical":"SBrkr","1stFlrSF":896,"2ndFlrSF":0,"LowQualFinSF":0,"GrLivArea":896,"BsmtFullBath":0.0,"BsmtHalfBath":0.0,"FullBath":1,"HalfBath":0,"BedroomAbvGr":2,"KitchenAbvGr":1,"KitchenQual":"TA","TotRmsAbvGrd":5,"Functional":"Typ","Fireplaces":0,"FireplaceQu":"Gd","GarageType":"Attchd","GarageYrBlt":1961.0,"GarageFinish":"Unf","GarageCars":1.0,"GarageArea":730.0,"GarageQual":"TA","GarageCond":"TA","PavedDrive":"Y","WoodDeckSF":140,"OpenPorchSF":0,"EnclosedPorch":0,"3SsnPorch":0,"ScreenPorch":120,"PoolArea":0,"PoolQC":"Gd","Fence":"MnPrv","MiscFeature":"Shed","MiscVal":0,"MoSold":6,"YrSold":2010,"SaleType":"WD","SaleCondition":"Normal"}]
    x = pd.DataFrame.from_dict(json_data)
    prediction = predict(x, model)
    return {
        'status': 200,
        'prediction': prediction,
    }

