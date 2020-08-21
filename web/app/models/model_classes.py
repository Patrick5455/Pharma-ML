from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, train_test_split, GridSearchCV, \
    RandomizedSearchCV


class Preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, data, y=None):
        return self

    def check(self, row):
        if isinstance(row['PromoInterval'], str) and row['month_str'] in row['PromoInterval']:
            return 1
        else:
            return 0

    def transform(self, data):
        self.cols = data.columns
        data = data.sort_values(['Date'], ascending=False)
        mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
        data.StoreType.replace(mappings, inplace=True)
        data.Assortment.replace(mappings, inplace=True)
        data.StateHoliday.replace(mappings, inplace=True)

        # extract some features from date column
        data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
        data['Month'] = data.Date.dt.month
        data['Year'] = data.Date.dt.year
        data['Day'] = data.Date.dt.day
        data['WeekOfYear'] = data.Date.dt.weekofyear

        # calculate competitor open time in months
        data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + \
                                  (data.Month - data.CompetitionOpenSinceMonth)
        data['CompetitionOpen'] = data['CompetitionOpen'].apply(lambda x: x if x > 0 else 0)

        # calculate promo2 open time in months
        data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + \
                            (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
        data['PromoOpen'] = data['PromoOpen'].apply(lambda x: x if x > 0 else 0)

        # Indicate whether the month is in promo interval
        month2str = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', \
                     7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        data['month_str'] = data.Month.map(month2str)
        if 'Sales' in self.cols:
            data = data[(data.Open != 0) & (data.Sales > 0)]
        data['IsPromoMonth'] = data.apply(lambda row: self.check(row), axis=1)

        # select the features we need
        features = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
                    'StoreType', 'Assortment', 'CompetitionDistance',
                    'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
                    'Promo2SinceWeek', 'Promo2SinceYear', 'Year', 'Month', 'Day',
                    'WeekOfYear', 'CompetitionOpen', 'PromoOpen', 'IsPromoMonth']

        data = data[features]

        return data


class Regressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        x_train_total, y_train_total = X, np.log1p(y)
        dtrain = xgb.DMatrix(x_train_total, y_train_total)
        # specify parameters via map
        params = {"objective": "reg:linear",  # for linear regression
                  "booster": "gbtree",  # use tree based models
                  "eta": 0.03,  # learning rate
                  "max_depth": 10,  # maximum depth of a tree
                  "subsample": 0.9,  # Subsample ratio of the training instances
                  "colsample_bytree": 0.7,  # Subsample ratio of columns when constructing each tree
                  "silent": 1,  # silent mode
                  "seed": 10  # Random number seed
                  }
        num_round = 1000
        self.model = xgb.train(params, dtrain, num_round)
        return self

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
