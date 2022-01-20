import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import numpy as np


class Data:

    def __init__(self):
        self.bad_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
        self.better_feature = ['LotArea', 'LowQualFinSF', 'OverallQual', 'ExterQual', 'Condition1',
                               'LotShape', 'ExterCond', 'Functional', 'EnclosedPorch', 'GarageYrBlt',
                               'Condition2', 'MSSubClass', 'MasVnrArea', 'YearRemodAdd', 'LandContour',
                               'PoolArea', 'Exterior2nd', 'MSZoning', 'BldgType', 'GarageType',
                               'PavedDrive', 'MasVnrType', 'LotFrontage', 'KitchenQual', 'BsmtQual',
                               'LotConfig', 'GarageArea', 'KitchenAbvGr', 'TotalBsmtSF',
                               'Neighborhood', 'Street', 'ScreenPorch', 'Foundation', 'TotRmsAbvGrd',
                               'SaleCondition', 'Utilities', 'RoofStyle', 'FullBath', 'WoodDeckSF',
                               'RoofMatl', 'HeatingQC', 'SaleType', 'GarageCars', 'OpenPorchSF',
                               'YrSold', 'BsmtHalfBath']
        self.numerical_imputer = SimpleImputer(strategy='constant')
        self.object_imputer = SimpleImputer(strategy='most_frequent')
        self.ordinal = OrdinalEncoder(handle_unknown='error')

    def get_data(self, df, fit=True):
        # Drop bad features
        df.drop(self.bad_features, axis=1, inplace=True)

        # Imputer mising values
        df_object = df.select_dtypes('object')
        df_num = df.select_dtypes('number')

        if fit:
            df_num_imp = pd.DataFrame(self.numerical_imputer.fit_transform(df_num), columns=df_num.columns)
            df_object_imp = pd.DataFrame(self.object_imputer.fit_transform(df_object), columns=df_object.columns)
            df_object_ord = pd.DataFrame(self.ordinal.fit_transform(df_object_imp), columns=df_object_imp.columns)
        else:
            df_num_imp = pd.DataFrame(self.numerical_imputer.transform(df_num), columns=df_num.columns)
            df_object_imp = pd.DataFrame(self.object_imputer.transform(df_object), columns=df_object.columns)
            df_object_ord = pd.DataFrame(self.ordinal.transform(df_object_imp), columns=df_object_imp.columns)

        df_ = pd.concat([df_num_imp, df_object_ord], join='inner', axis=1)

        return df_


class Data2:

    def __init__(self):

        LotShape = ['IR3', 'IR2', 'IR1', 'Reg']
        Utilities = ['ELO', 'NoSeWa', 'NoSewr', 'AllPub']
        LandSlope = ['Sev', 'Mod', 'Gtl']
        HouseStyle = ['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl']  # posible oh
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

        self.ordinal_cols_categories = [LotShape, Utilities, LandSlope, HouseStyle, ExterQual, ExterCond, BsmtQual,
                                        BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, HeatingQC, CentralAir,
                                        Electrical, KitchenQual, Functional, FireplaceQu, GarageType, GarageFinish,
                                        GarageQual, GarageCond, PavedDrive, PoolQC, Fence]

        self.col_fill_NA = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                            'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                            'PoolQC', 'Fence', 'MiscFeature']
        self.col_fill_0 = ['LotFrontage', 'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF2', 'GarageArea',
                           'BsmtFinSF1', 'GarageCars', 'TotalBsmtSF', 'BsmtUnfSF']
        self.col_fill_mf = ['MSZoning', 'Electrical', 'GarageYrBlt', 'Functional', 'Utilities', 'Exterior2nd',
                            'Exterior1st', 'SaleType', 'KitchenQual']

        self.ordinal_cols = ['LotShape', 'Utilities', 'LandSlope', 'HouseStyle', 'ExterQual', 'ExterCond', 'BsmtQual',
                             'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir',
                             'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
                             'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']

        self.oh_cols = ['MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1',
                        'Condition2', 'BldgType', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                        'Foundation', 'Heating', 'MiscFeature', 'SaleType', 'SaleCondition']

        self.imputer_N = SimpleImputer(strategy='constant', fill_value='NA')
        self.imputer_0 = SimpleImputer(strategy='constant', fill_value=0)
        self.imputer_mf = SimpleImputer(strategy='most_frequent')
        self.ordinal = OrdinalEncoder(categories=self.ordinal_cols_categories)
        self.oh_encoder = OneHotEncoder(handle_unknown='ignore')

    def get_data2(self, df, fit=True):
        col_fill_NA = self.col_fill_NA
        col_fill_0 = self.col_fill_0
        col_fill_mf = self.col_fill_mf
        ordinal_cols = self.ordinal_cols
        if fit:
            df[col_fill_NA] = pd.DataFrame(self.imputer_N.fit_transform(df[col_fill_NA]), columns=col_fill_NA)
            df[col_fill_0] = pd.DataFrame(self.imputer_0.fit_transform(df[col_fill_0]), columns=col_fill_0)
            df[col_fill_mf] = pd.DataFrame(self.imputer_mf.fit_transform(df[col_fill_mf]), columns=col_fill_mf)
            df['MasVnrType'].replace('None', 'NA', inplace=True)
            df[ordinal_cols] = pd.DataFrame(self.ordinal.fit_transform(df[ordinal_cols]), columns=ordinal_cols)
            df_oh = pd.DataFrame(self.oh_encoder.fit_transform(df[self.oh_cols]).toarray())
        else:
            df[col_fill_NA] = pd.DataFrame(self.imputer_N.transform(df[col_fill_NA]), columns=col_fill_NA)
            df[col_fill_0] = pd.DataFrame(self.imputer_0.transform(df[col_fill_0]), columns=col_fill_0)
            df[col_fill_mf] = pd.DataFrame(self.imputer_mf.transform(df[col_fill_mf]), columns=col_fill_mf)
            df['MasVnrType'].replace('None', 'NA', inplace=True)
            df[ordinal_cols] = pd.DataFrame(self.ordinal.transform(df[ordinal_cols]), columns=ordinal_cols)
            df_oh = pd.DataFrame(self.oh_encoder.transform(df[self.oh_cols]).toarray())

        df.drop(self.oh_cols, axis=True, inplace=True)
        df = pd.concat([df, df_oh], join='inner', axis=1)
        df.drop('Id', axis=1, inplace=True)

        return df


class DataPrep:

    def __init__(self):

        LotShape = ['IR3', 'IR2', 'IR1', 'Reg']
        Utilities = ['ELO', 'NoSeWa', 'NoSewr', 'AllPub']
        LandSlope = ['Sev', 'Mod', 'Gtl']
        HouseStyle = ['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl']  # posible oh
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

        self.high_skew = ['MiscVal', 'PoolArea', 'LotArea', 'LowQualFinSF', '3SsnPorch',
                          'KitchenAbvGr', 'BsmtFinSF2', 'EnclosedPorch', 'ScreenPorch',
                          'BsmtHalfBath', 'OpenPorchSF', 'WoodDeckSF', '1stFlrSF', 'BsmtFinSF1',
                          'MSSubClass', 'GrLivArea', 'TotalBsmtSF', 'BsmtUnfSF', '2ndFlrSF',
                          'TotRmsAbvGrd', 'Fireplaces', 'HalfBath', 'BsmtFullBath', 'OverallCond',
                          'YearBuilt']

        self.ordinal_cols_categories = [LotShape, Utilities, LandSlope, HouseStyle, ExterQual, ExterCond, BsmtQual,
                                        BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, HeatingQC, CentralAir,
                                        Electrical, KitchenQual, Functional, FireplaceQu, GarageType, GarageFinish,
                                        GarageQual, GarageCond, PavedDrive, PoolQC, Fence]

        self.col_fill_NA = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                            'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                            'PoolQC', 'Fence', 'MiscFeature']
        self.col_fill_0 = ['LotFrontage', 'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF2', 'GarageArea',
                           'BsmtFinSF1', 'GarageCars', 'TotalBsmtSF', 'BsmtUnfSF']
        self.col_fill_mf = ['MSZoning', 'Electrical', 'GarageYrBlt', 'Functional', 'Utilities', 'Exterior2nd',
                            'Exterior1st', 'SaleType', 'KitchenQual']

        self.ordinal_cols = ['LotShape', 'Utilities', 'LandSlope', 'HouseStyle', 'ExterQual', 'ExterCond', 'BsmtQual',
                             'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir',
                             'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
                             'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']

        self.oh_cols = ['MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1',
                        'Condition2', 'BldgType', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                        'Foundation', 'Heating', 'MiscFeature', 'SaleType', 'SaleCondition']

        self.imputer_N = SimpleImputer(strategy='constant', fill_value='NA')
        self.imputer_0 = SimpleImputer(strategy='constant', fill_value=0)
        self.imputer_mf = SimpleImputer(strategy='most_frequent')
        self.ordinal = OrdinalEncoder(categories=self.ordinal_cols_categories)
        self.oh_encoder = OneHotEncoder(handle_unknown='ignore')

    def get_data(self, df, y=None, fit=True):

        col_fill_NA = self.col_fill_NA
        col_fill_0 = self.col_fill_0
        col_fill_mf = self.col_fill_mf
        ordinal_cols = self.ordinal_cols

        if fit:

            df[col_fill_NA] = pd.DataFrame(self.imputer_N.fit_transform(df[col_fill_NA]), columns=col_fill_NA)
            df[col_fill_0] = pd.DataFrame(self.imputer_0.fit_transform(df[col_fill_0]), columns=col_fill_0)
            df[col_fill_mf] = pd.DataFrame(self.imputer_mf.fit_transform(df[col_fill_mf]), columns=col_fill_mf)
            df['MasVnrType'].replace('None', 'NA', inplace=True)
            df[ordinal_cols] = pd.DataFrame(self.ordinal.fit_transform(df[ordinal_cols]), columns=ordinal_cols)
            df_oh = pd.DataFrame(self.oh_encoder.fit_transform(df[self.oh_cols]).toarray())
        else:
            df[col_fill_NA] = pd.DataFrame(self.imputer_N.transform(df[col_fill_NA]), columns=col_fill_NA)
            df[col_fill_0] = pd.DataFrame(self.imputer_0.transform(df[col_fill_0]), columns=col_fill_0)
            df[col_fill_mf] = pd.DataFrame(self.imputer_mf.transform(df[col_fill_mf]), columns=col_fill_mf)
            df['MasVnrType'].replace('None', 'NA', inplace=True)
            df[ordinal_cols] = pd.DataFrame(self.ordinal.transform(df[ordinal_cols]), columns=ordinal_cols)
            df_oh = pd.DataFrame(self.oh_encoder.transform(df[self.oh_cols]).toarray())

        df.drop(self.oh_cols, axis=True, inplace=True)
        df = pd.concat([df, df_oh], join='inner', axis=1)
        df.drop('Id', axis=1, inplace=True)

        df['GarageYrBlt'] = df['GarageYrBlt'].astype('float')

        for ft in self.high_skew:
            df[ft] = np.log1p(df[ft])

        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

        if y is not None:
            y = np.log1p(y)

        return df, y
