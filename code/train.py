import pandas as pd
import joblib
import gzip
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from mlxtend.regressor import StackingCVRegressor
from sklearn.metrics import ConfusionMatrixDisplay
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# Defineing necesaries variales columns
LotShape = ['IR3', 'IR2', 'IR1', 'Reg']
Utilities = ['ELO', 'NoSeWa', 'NoSewr', 'AllPub']
LandSlope = ['Sev', 'Mod', 'Gtl']
HouseStyle = ['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl']
ExterQual = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
ExterCond = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
BsmtQual = ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
BsmtCond = ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
BsmtExposure = ['NA', 'No', 'Mn', 'Av', 'Gd']
BsmtFinType1 = ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
BsmtFinType2 = ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
HeatingQC = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
CentralAir = ['N', 'Y']
Electrical = ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr']
KitchenQual = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
Functional = ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ']
FireplaceQu = ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
GarageType = ['NA', 'Detchd', 'CarPort', 'BuiltIn', 'Basment', 'Attchd', '2Types']
GarageFinish = ['NA', 'Unf', 'RFn', 'Fin']
GarageQual = ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
GarageCond = ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
PavedDrive = ['N', 'P', 'Y']
PoolQC = ['NA', 'Fa', 'TA', 'Gd', 'Ex']
Fence = ['NA', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']
ord_cols_mf_cat = [Electrical, Functional, Utilities, KitchenQual]
ord_cols_na_cat = [LotShape, LandSlope, HouseStyle, ExterQual, ExterCond, BsmtQual,
                   BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, HeatingQC,
                   CentralAir, FireplaceQu, GarageType, GarageFinish, GarageQual,
                   GarageCond, PavedDrive, PoolQC, Fence]
num_cols_0 = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
              'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
              'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
              'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
              'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
              'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
              'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
              'MoSold', 'YrSold']
ord_cols_mf = ['Electrical', 'Functional', 'Utilities', 'KitchenQual']
ord_cols_na = ['LotShape', 'LandSlope', 'HouseStyle', 'ExterQual', 'ExterCond',
               'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
               'HeatingQC', 'CentralAir', 'FireplaceQu', 'GarageType', 'GarageFinish',
               'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']
oh_cols_na = ['MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood',
              'Condition1', 'Condition2', 'BldgType', 'RoofStyle', 'RoofMatl',
              'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
              'MiscFeature', 'SaleType', 'SaleCondition']

# Importamos datos
df = pd.read_csv('in/train.csv')
y = df['SalePrice']
df.drop('SalePrice', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.15, random_state=7)

# Define Pipelines
pipe_num_0 = Pipeline(steps=[
    ('imp', SimpleImputer(strategy='constant', fill_value=0))
    ])
pipe_ord_mf = Pipeline(steps=[
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('ord', OrdinalEncoder(categories=ord_cols_mf_cat))
    ])
pipe_ord_na = Pipeline(steps=[
    ('imp', SimpleImputer(strategy='constant', fill_value='NA')),
    ('ord', OrdinalEncoder(categories=ord_cols_na_cat))
    ])
pipe_oh_na = Pipeline(steps=[
    ('imp', SimpleImputer(strategy='constant', fill_value='NA')),
    ('ord', OneHotEncoder(handle_unknown='ignore'))
    ])

# Preprocessing all columns
preprocessing = ColumnTransformer(transformers=[('drop_id', 'drop', ['Id']),
                                                ('trans_num_0', pipe_num_0, num_cols_0),
                                                ('trans_ord_mf', pipe_ord_mf, ord_cols_mf),
                                                ('trans_ord_na', pipe_ord_na, ord_cols_na),
                                                ('trans_oh_na', pipe_oh_na, oh_cols_na)
                                                ])

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

# declarando models
ridge = RidgeCV(alphas=alphas_alt, cv=kfolds)

lasso = LassoCV(max_iter=100, alphas=alphas2,
                random_state=42, cv=kfolds)

elasticnet = ElasticNetCV(max_iter=100, alphas=e_alphas,
                          cv=kfolds, l1_ratio=e_l1ratio)

gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10,
                                loss='huber', random_state=42)

lightgbm = LGBMRegressor(objective='regression',
                         num_leaves=4,
                         learning_rate=0.01,
                         n_estimators=5000,
                         max_bin=200,
                         bagging_fraction=0.75,
                         bagging_freq=5,
                         bagging_seed=7,
                         feature_fraction=0.2,
                         feature_fraction_seed=7,
                         verbose=-1,
                         # min_data_in_leaf=2,
                         # min_sum_hessian_in_leaf=11
                         )

xgboost = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                       max_depth=3, min_child_weight=0,
                       gamma=0, subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:squarederror', nthread=-1,
                       scale_pos_weight=1, seed=27,
                       reg_alpha=0.00006)

# Stack model
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet,
                                            gbr, xgboost, lightgbm),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)

# Voting model
voting = VotingRegressor(estimators=[('ridge',ridge), ('lasso',lasso), ('elasticnet',elasticnet),
                                     ('gbr',gbr), ('lightgbm',lightgbm), 
                                     ('xgboost',xgboost),('stack_gen',stack_gen)],
                        weights=[0.1,0.1,0.1,0.1,0.1,0.15,0.25])

# Finaly pipeline
pipe = Pipeline(steps=[('preprocessing', preprocessing),
                       ('models', voting)
                       ])

# Fit pipeline
print('fitting')
pipe.fit(X_train, y_train)

# Test accuracy
print("Accuracy: %s " % str(ridge.score(X_test, y_test)))

# Export model pipeline
joblib.dump(ridge, gzip.open('../models/model_binary.dat.gz', "wb"))
