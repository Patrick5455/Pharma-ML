#!/usr/bin/env python
# coding: utf-8

<<<<<<< HEAD
# In[22]:


import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle

# predefined modules
# from prediction_utilities import preprocess
# from prediction_utilities import preprocess


#preprocessing libraries 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer


import pandas as pd
import seaborn as sns
import numpy as np


# In[23]:
=======
# In[1]:
>>>>>>> fef695eaa11f3a653a112b9933845269faaceb0d


def outlier_vars(data, show_plot=False):
    
    """
    This functions checks for columns with outlers using the IQR method
    
    It accespts as argmuent a dataset. 
    show_plot can be set to True to output pairplots of outlier columns
    
    """
    
    outliers = [] 
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    num_data = data.select_dtypes(include='number')
    result = dict ((((num_data < (Q1 - 1.5 * IQR)) | (num_data > (Q3 + 1.5 * IQR)))==True).any())
    for k,v in result.items():
        if v == True: 
            outliers.append(k)
    if show_plot:
        pair_plot = sns.pairplot(data[outliers]);
        print(f'{result},\n\n Visualization of outlier columns')
        return pair_plot
    else:
        return data[outliers]


<<<<<<< HEAD
# In[3]:


def preprocess(data, to_drop=[], save_path='', obj_name='prcsd_data.pkl'):
=======
<<<<<<< HEAD
# In[24]:


def preprocess(data, to_drop=[], save_path='', obj_name='prcsd_data.pkl'):
=======
# In[2]:


def preprocess(data, to_drop=[]):
>>>>>>> fef695eaa11f3a653a112b9933845269faaceb0d
>>>>>>> wip
    
    """
    The preprocess function takes as primary argument the data 
    and peform the following stepwise transformations to it:
    
<<<<<<< HEAD
    1. impute missing val
    ues of numerical and categorical columns 
=======
<<<<<<< HEAD
    1. impute missing val
    ues of numerical and categorical columns 
=======
    1. impute missing values of numerical and categorical columns 
>>>>>>> fef695eaa11f3a653a112b9933845269faaceb0d
>>>>>>> wip
    using median and constant values respectively
    
    2. scales dataset using the RobustScaler (robust to outlier values present in this dataset)
    
    3. Encodes categorical values to numerical values
    """
    
    columns = data.columns.to_list()
    
    # split data to numeric vs categorical
    numeric_features = data.select_dtypes(include=[
        'int64', 'float64']).columns
    
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> wip
    
    
    if len(to_drop) > 0:
        data = data.drop(to_drop, axis=1)
        categorical_features = data.select_dtypes(include=[
        'object']).columns
        #print(categorical_features) 
<<<<<<< HEAD
=======
=======
    if len(to_drop) > 0:
        categorical_features = data.select_dtypes(include=[
        'object']).drop(to_drop, axis=1).columns
        print(categorical_features)
>>>>>>> fef695eaa11f3a653a112b9933845269faaceb0d
>>>>>>> wip
    else: 
        categorical_features = data.select_dtypes(include=[
        'object']).columns
        
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> wip
    numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())])

    categorical_transformer = Pipeline(steps=[
<<<<<<< HEAD
    ('imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
=======
    #('imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan)),
        ('onehot', LabelEncoder())
>>>>>>> wip
    ])
    
    
    # Bundle preprocessing for numerical and categorical data
<<<<<<< HEAD
=======
=======
    categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing'))])
    
    numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
    ])
    # missing_values = np.nan
    
# Bundle preprocessing for numerical and categorical data
>>>>>>> fef695eaa11f3a653a112b9933845269faaceb0d
>>>>>>> wip
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor) ])
    
    for col in to_drop:
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> wip
        columns.remove(col) 
    
    trans_data = my_pipeline.fit_transform(data)
    
    if save_path:
        pickle_out = open(save_path+obj_name, 'wb')
        pickle.dump(trans_data, pickle_out)
        pickle_out.close()
    
    return pd.DataFrame(trans_data, columns=columns)


# In[25]:


def model_pipelines(preprocessor,model_algos=[]):
    for algo in model_algos:
        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', algo)])
        pipe.fit(X_train, y_train)   
        print(classifier)
        print("model score: %.3f" % pipe.score(X_test, y_test))


# In[26]:


def change_datatype(data, dtype, col_list, date_col_name='Date'):
    """
    This converts specified columns names of a data to the specified data type
    It also uses the name od the date column of the dataset to know the data columns
    ans convert to datetime object
    """
    for i in col_list:
        if i != date_col_name:
            data = data.astype({i:dtype})
        else:
            data[date_col_name] = pd.to_datetime(data.Date)
    return data


# In[27]:


def save_load_model (action, model=None, model_name='new_model.pickle',
                          path = ''):
    
    if action == "wb":
        pickle_out = open(path+model_name,action)
        pickle.dump(model, pickle_out)
        pickle_out.close() 
    
    elif action=="rb":
        pickle_in = open(path+model_name,action)
        model=pickle.load(pickle_in)
    
    return model


# In[28]:


def fill_na(data, num_type='median'):
    
    
    """
    Fill categorical clumns containing missing vlaues with mode
    
    Fill numerical  clumns containing missing vlaues with specified type
    
    median is default
    
    """
    cat = categorical_features = data.select_dtypes(include=[
        'object']).columns
    
    num = numeric_features = data.select_dtypes(include=[
        'int64', 'float64']).columns
    
    for col in cat:
        data[col].fillna(data[col].mode()[0], inplace=True) 
        

    if num_type == "median":
        
        for col in num:
            data[col].fillna(data[col].median(), inplace=True)
    
    if num_type == "mean":
        
        for col in num:
            data[col].fillna(data[col].mean(), inplace=True)
            
    if num_type == "mode":
        
        for col in num:
            data[col].fillna(data[col].mode()[0], inplace=True)
        
    return data
<<<<<<< HEAD
=======


# In[29]:


def extract_dates(data, date_col_name='Date', season=False):
    
    """
    Extracts various date types from a date column
    
    If season is specified, adds column for season
    """
    
    data["Month"] = data[date_col_name].dt.month
    data["Quarter"] = data[date_col_name].dt.quarter
    data["Year"] = data[date_col_name].dt.year
    data["Day"] = data[date_col_name].dt.day
    data["Week"] = data[date_col_name].dt.week
    
    if season:
        data["Season"] = np.where(data["Month"].isin([3,4,5]),"Spring",
        np.where(data["Month"].isin([6,7,8]),"Summer",np.where(data["Month"].isin
        ([9,10,11]),"Fall",np.where(data["Month"].isin([12,1,2]),"Winter","None"))))
        
        
    return data


# In[ ]:



=======
        columns.remove(col)
    print('Hello')
    
    trans_data = my_pipeline.fit_transform(data)
    return pd.DataFrame( trans_data, columns=columns) 
>>>>>>> fef695eaa11f3a653a112b9933845269faaceb0d
>>>>>>> wip


# In[ ]:




