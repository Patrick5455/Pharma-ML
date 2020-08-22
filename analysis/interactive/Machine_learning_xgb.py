#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline,make_pipeline 
from sklearn.metrics import mean_squared_error
from hypopt import GridSearch
import xgboost as xgb
import pickle
from sklearn.base import BaseEstimator, TransformerMixin,RegressorMixin
from sklearn.model_selection import cross_val_score,cross_val_predict, KFold,train_test_split,GridSearchCV,RandomizedSearchCV


# In[2]:


na_value=['',' ','nan','Nan','NaN','na']
train=pd.read_csv('./train.csv',na_values=na_value)
store=pd.read_csv('./store.csv',na_values=na_value)
test=pd.read_csv('./test.csv',na_values=na_value)


# In[3]:


# fillna in store with 0 has better result than median()
store.fillna(0, inplace=True)
test.fillna(value = 1, inplace = True)


# In[4]:


train = pd.merge(train, store, on='Store')
test = pd.merge(test, store, on='Store')


# In[22]:


# split the last 6 weeks data as hold-out set (idea from Gert https://www.kaggle.com/c/rossmann-store-sales/discussion/18024)
train = train.sort_values(['Date'],ascending = False)
train_total = train.copy()

split_index = 6*7*1115
valid = train[:split_index] 
train = train[split_index:]

# only use data of Sales>0 and Open is 1
valid = valid[(valid.Open != 0)&(valid.Sales >0)]
train = train[(train.Open != 0)&(train.Sales >0)]
train_total = train_total[(train_total.Open != 0)&(train_total.Sales >0)]


# In[23]:


train.StoreType.unique()


# In[24]:


def process(data, isTest = False):
    # label encode some features
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)
    
    # extract some features from date column  
    data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
    data['Month'] = data.Date.dt.month
    data['Year'] = data.Date.dt.year
    data['Day'] = data.Date.dt.day
    data['WeekOfYear'] = data.Date.dt.weekofyear
    
    # calculate competiter open time in months
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) +         (data.Month - data.CompetitionOpenSinceMonth)
    data['CompetitionOpen'] = data['CompetitionOpen'].apply(lambda x: x if x > 0 else 0)
    
    # calculate promo2 open time in months
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) +         (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data['PromoOpen'].apply(lambda x: x if x > 0 else 0)
                                                 
    # Indicate whether the month is in promo interval
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',              7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
    data['month_str'] = data.Month.map(month2str)

    def check(row):
        if isinstance(row['PromoInterval'],str) and row['month_str'] in row['PromoInterval']:
            return 1
        else:
            return 0
        
    data['IsPromoMonth'] =  data.apply(lambda row: check(row),axis=1)    
    
    # select the features we need
    features = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
       'StoreType', 'Assortment', 'CompetitionDistance',
       'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
       'Promo2SinceWeek', 'Promo2SinceYear', 'Year', 'Month', 'Day',
       'WeekOfYear', 'CompetitionOpen', 'PromoOpen', 'IsPromoMonth']  
    if not isTest:
        features.append('Sales')
        
    data = data[features]
    return data

train = process(train)
valid = process(valid)
train_total = process(train_total)
x_test = process(test,isTest = True)


# In[25]:


# sort by index
valid.sort_index(inplace = True)
train.sort_index(inplace = True)
train_total.sort_index(inplace = True)

# split x and y
x_train, y_train = train.drop(columns = ['Sales']), np.log1p(train['Sales'])
x_valid, y_valid = valid.drop(columns = ['Sales']), np.log1p(valid['Sales'])
x_train_total, y_train_total = train_total.drop(columns = ['Sales']), np.log1p(train_total['Sales'])


# In[26]:


# define eval metrics
def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)


# In[ ]:





# In[27]:


clf = RandomForestRegressor(n_estimators = 15)
clf.fit(x_train, y_train)
# validation
y_pred = clf.predict(x_valid)
error = rmspe(np.expm1(y_valid), np.expm1(y_pred))
print('RMSPE: {:.4f}'.format(error))


# In[37]:


def plot_feature_importances(model,name):
    feature=pd.DataFrame({'imp':model.feature_importances_,'features':x_train.columns})
    feature=feature.sort_values(by='imp',ascending = True).head(10)
    n_features =feature.shape[0]
    plt.barh(range(n_features),feature.imp, align='center')
    plt.yticks(np.arange(n_features), x_train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.title('Feature importance in Tree model')
    plt.savefig(name+'.jpg')
    return feature.features.values
features=plot_feature_importances(clf,'feature_importance')


# In[29]:


params = {"objective": "reg:linear", # for linear regression
          "booster" : "gbtree",   # use tree based models 
          "eta": 0.03,   # learning rate
          "max_depth": 10,    # maximum depth of a tree
          "subsample": 0.9,    # Subsample ratio of the training instances
          "colsample_bytree": 0.7,   # Subsample ratio of columns when constructing each tree
          "silent": 1,   # silent mode
          "seed": 10   # Random number seed
          }
num_boost_round = 4000

dtrain = xgb.DMatrix(x_train, y_train)
dvalid = xgb.DMatrix(x_valid, y_valid)
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
# train the xgboost model
model = xgb.train(params, dtrain, num_boost_round, evals=watchlist,   early_stopping_rounds= 100, feval=rmspe_xg, verbose_eval=True)


# In[30]:


y_pred = model.predict(xgb.DMatrix(x_valid))
error = rmspe(np.expm1(y_valid), np.expm1(y_pred))
print('RMSPE: {:.4f}'.format(error))


# In[ ]:





# In[31]:


def correction():
    weights = np.arange(0.98, 1.02, 0.005)
    errors = []
    for w in weights:
        error = rmspe(np.expm1(y_valid), np.expm1(y_pred*w))
        errors.append(error)
        
    # make line plot
    plt.plot(weights, errors)
    plt.xlabel('weight')
    plt.ylabel('RMSPE')
    plt.title('RMSPE Curve')
    # print min error
    idx = errors.index(min(errors))
    print('Best weight is {}, RMSPE is {:.4f}'.format(weights[idx], min(errors)))
    
correction()


# In[32]:


x_train_total.head().append(x_train_total.tail())


# In[38]:


dtrain = xgb.DMatrix(x_train_total, y_train_total)
dtest = xgb.DMatrix(x_test)
# specify parameters via map
params = {"objective": "reg:linear", # for linear regression
          "booster" : "gbtree",   # use tree based models 
          "eta": 0.03,   # learning rate
          "max_depth": 10,    # maximum depth of a tree
          "subsample": 0.9,    # Subsample ratio of the training instances
          "colsample_bytree": 0.7,   # Subsample ratio of columns when constructing each tree
          "silent": 1,   # silent mode
          "seed": 10   # Random number seed
          }
num_round = 1000
model = xgb.train(params, dtrain, num_round)
# make prediction
preds = model.predict(dtest)


# In[39]:


# plot feature importance, show top 10 features
fig, ax = plt.subplots(figsize=(8,8))
xgb.plot_importance(model, max_num_features= 10, height=0.5, ax=ax)
plt.show()


# In[ ]:


result = pd.DataFrame({"Id": test["Id"],'Sales': np.expm1(preds*0.995)})


# Multiply the exponent of the predicted values with 0.995 

# In[13]:


##One transformer for all
class Preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, data, y=None):
        return self
    def check(self,row):
            if isinstance(row['PromoInterval'],str) and row['month_str'] in row['PromoInterval']:
                return 1
            else:
                return 0

    def transform(self,data):
        self.cols=data.columns
        data = data.sort_values(['Date'],ascending = False)
        mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
        data.StoreType.replace(mappings, inplace=True)
        data.Assortment.replace(mappings, inplace=True)
        data.StateHoliday.replace(mappings, inplace=True)
    
        # extract some features from date column  
        data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
        data['Month'] = data.Date.dt.month
        data['Year'] = data.Date.dt.year
        data['Day'] = data.Date.dt.day
        data['WeekOfYear'] = data.Date.dt.weekofyear
    
        # calculate competiter open time in months
        data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) +             (data.Month - data.CompetitionOpenSinceMonth)
        data['CompetitionOpen'] = data['CompetitionOpen'].apply(lambda x: x if x > 0 else 0)
    
        # calculate promo2 open time in months
        data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) +             (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
        data['PromoOpen'] = data['PromoOpen'].apply(lambda x: x if x > 0 else 0)
                                                 
        # Indicate whether the month is in promo interval
        month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',                  7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
        data['month_str'] = data.Month.map(month2str)
        if 'Sales' in self.cols:
            data = data[(data.Open != 0)&(data.Sales >0)]
        data['IsPromoMonth'] =  data.apply(lambda row: self.check(row),axis=1)    
    
        # select the features we need
        features = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
           'StoreType', 'Assortment', 'CompetitionDistance',
           'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
           'Promo2SinceWeek', 'Promo2SinceYear', 'Year', 'Month', 'Day',
           'WeekOfYear', 'CompetitionOpen', 'PromoOpen', 'IsPromoMonth']  
        
        
        data = data[features]

        return data
        
        
        
        
       


# In[24]:


class Regressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        x_train_total, y_train_total = X, np.log1p(y)
        dtrain = xgb.DMatrix(x_train_total, y_train_total)
        # specify parameters via map
        params = {"objective": "reg:linear", # for linear regression
                  "booster" : "gbtree",   # use tree based models 
                  "eta": 0.03,   # learning rate
                  "max_depth": 10,    # maximum depth of a tree
                    "subsample": 0.9,    # Subsample ratio of the training instances
                    "colsample_bytree": 0.7,   # Subsample ratio of columns when constructing each tree
                  "silent": 1,   # silent mode
                  "seed": 10   # Random number seed
          }
        num_round = 1000
        self.model = xgb.train(params, dtrain, num_round)
        return self
    
    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)


# In[25]:


X,y=train.drop(['Sales'],axis=1),train['Sales']
pipe = Pipeline(steps=[('preprocesssing',Preprocessing()),('Reg',Regressor())])
pipe.fit(X,y)


# In[27]:


pickle.dump(pipe, open('21-08-2020-16-32-31-00.pkl','wb'))


# In[20]:


model = pickle.load(open('model.pkl','rb'))


# In[28]:


np.expm1(0.995*pipe.predict(test)) # Output should be processed to obtain original values


# In[ ]:




