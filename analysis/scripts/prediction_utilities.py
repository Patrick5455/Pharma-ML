#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


def preprocess(data, to_drop=[]):
    
    """
    The preprocess function takes as primary argument the data 
    and peform the following stepwise transformations to it:
    
    1. impute missing values of numerical and categorical columns 
    using median and constant values respectively
    
    2. scales dataset using the RobustScaler (robust to outlier values present in this dataset)
    
    3. Encodes categorical values to numerical values
    """
    
    columns = data.columns.to_list()
    
    # split data to numeric vs categorical
    numeric_features = data.select_dtypes(include=[
        'int64', 'float64']).columns
    
    if len(to_drop) > 0:
        categorical_features = data.select_dtypes(include=[
        'object']).drop(to_drop, axis=1).columns
        print(categorical_features)
    else: 
        categorical_features = data.select_dtypes(include=[
        'object']).columns
        
    categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing'))])
    
    numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
    ])
    # missing_values = np.nan
    
# Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor) ])
    
    for col in to_drop:
        columns.remove(col)
    print('Hello')
    
    trans_data = my_pipeline.fit_transform(data)
    return trans_data#pd.DataFrame(#, columns=columns) 


# In[ ]:




