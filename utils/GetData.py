import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


class Data:

    def __init__(self):
        self.bad_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
    
    def get_data(self, df):
        # Drop bad features
        df.drop(self.bad_features, axis=1, inplace=True)

        # Imputer mising values
        df_object = df.select_dtypes('object')
        df_num  = df.select_dtypes('number')

        numerical_imputer = SimpleImputer(strategy='constant')
        object_imputer = SimpleImputer(strategy='most_frequent')
        ordinal = OrdinalEncoder(handle_unknown='error')

        df_num_imp = pd.DataFrame(numerical_imputer.fit_transform(df_num), columns= df_num.columns)
        df_object_imp = pd.DataFrame(object_imputer.fit_transform(df_object), columns= df_object.columns)
        df_object_ord = pd.DataFrame(ordinal.fit_transform(df_object_imp), columns=df_object_imp.columns)

        df_ = pd.concat([df_num_imp,df_object_ord], axis=1)

        return df_