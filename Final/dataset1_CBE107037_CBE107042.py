#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import randint
import statsmodels.api as sm


# In[2]:


def DealMissingData(X, dfX):
    ''' deal with missing data
        X: data, type(numpy array)
        dfX: same as X data, type(pandas array) '''
    
    print("----------- Start deal missing data -----------")
    TFarr = np.array(dfX.isna().any())
    for index, ele in enumerate(TFarr):
        if ele:
            if isinstance(X[0][index], str):
                # deal with string data
                imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                imputer.fit(X[:, [index]])
                X[:, [index]] = imputer.transform(X[:, [index]])
            else:
                # deal with digital data
                imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                imputer.fit(X[:, [index]])
                X[:, [index]] = imputer.transform(X[:, [index]])
            print(f"{index} column have missing data, fixed!")
        else:
            print(f"{index} column not have missing data")
            
    print("----------- End deal missing data! -----------")
    return X


# In[3]:


def MakeOneHot(X, column_name, pos={}):
    ''' make one-hot  
        X: data, type(numpy array)
        pos: where need to onehot, type(dictionary) '''
    
    print("----------- Start onehot -----------")
    FeaturesNum = len(X[0])    # init
    if bool(pos):
        # custom onehot (onehot pos that u want to)
        for key in pos:
            print(f"{FeaturesNum-pos[key]} column need to one-hot, fixed!")
            ct = ColumnTransformer([(key, OneHotEncoder(), [FeaturesNum-pos[key]])], remainder='passthrough')
            NewX = ct.fit_transform(X)
            X = NewX[:, 1:]
            FeaturesNum = len(X[0])
    else:
        # auto onehot (only onehot string cols)
        i = 0
        cn = list(column_name)
        while i < FeaturesNum:
            if isinstance(X[0][i], str):
                print(f"{i} column need to one-hot, fixed!")
                label = cn.pop(i)
                print(label)
                ct = ColumnTransformer([(str(i), OneHotEncoder(), [i])], remainder='passthrough')
                ct_X = ct.fit_transform(X)
                NewX = ct_X[:, 1:]
                OneHotLabel = ct.named_transformers_[str(i)]
                for num in range(len(OneHotLabel.categories_[0])-1):
                    cn.insert(num,label+str(num+1))
                i += len(NewX[0]) - len(X[0])
                X = NewX
                FeaturesNum = len(X[0])
            i += 1
    print("----------- End onehot -----------")
    
    return X,cn


# In[4]:


def NormalizeData(data_train, data_test):
    ''' normalize data
        data_train: training data, type(numpy array)
        data_test: testing data, type(numpy array) '''
    
    print("----------- Start normalize -----------")
    sc = StandardScaler()
    data_train = sc.fit_transform(data_train)
    data_test = sc.transform(data_test)
    
    print("----------- End normalize -----------")
    return data_train, data_test


# In[5]:


def DrawRelationship(X, y, label_x, label_y):
    for key in label_x:
        allarr = []
        for i in range(len(X)):
            allarr.append(X[i][label_x[key]])
        
        if ispred:
            plt.scatter(allarr, pred, c="blue")
        plt.scatter(allarr, y, c="red")
        plt.xlabel(key)
        plt.ylabel(label_y)
        plt.show() 


# In[6]:


def DrawPredict(gt, pred, title):
    ''' data [[groundTruth, predict], [], ....]'''
    
    data = []
    for i in range(len(gt)):
        data.append([gt[i], pred[i]])
        
    data.sort(key=lambda x:x[0])
    for index, ele in enumerate(data):
        plt.scatter(index, data[index][1], c="blue", s=0.7)
        plt.scatter(index, data[index][0], c="red", s=0.7)
    plt.xlabel("data sort by charges")
    plt.ylabel("charges")
    plt.title(title)
    plt.show() 


# In[7]:


def TrainAndTestModel(model, X_train, y_train, X_test, y_test, title):
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    train_acc = model.score(X_train, y_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_mae =  mean_absolute_error(y_train, train_pred)
    DrawPredict(y_train, train_pred, title+"_Train" + "\n" + f"RMSE: {train_rmse}, MAE: {train_mae}\nACC: {train_acc}")
    
    test_pred = model.predict(X_test)
    test_acc = model.score(X_test, y_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae =  mean_absolute_error(y_test, test_pred)
    DrawPredict(y_test, test_pred, title+"_Test" + "\n" + f"RMSE: {test_rmse}, MAE: {test_mae}\nACC: {test_acc}")
    return model


# <h1>Data preprocessing</h1>

# In[8]:


df = pd.read_csv("./DATA/final_project_dataset_1.csv")

dfX = df.iloc[:, :-1]
X = dfX.values
y = df.iloc[:, -1].values
column_name = np.array(dfX.columns.values)

#DrawRelationship(X, y, label_x={'age': 0, 'sex': 1, 'bmi': 2, 'children': 3, 'smoker': 4, 'region': 5}, label_y='charges')

X = DealMissingData(X, dfX)

#pos2onehot = {'Sex': 5, 'children': 3, 'Smoker': 2, 'Region': 1}
X,column_name = MakeOneHot(X,column_name)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# X_train, X_test = NormalizeData(X_train, X_test)
print(X[0])
for ele in zip(X[0], column_name):
    print(f"{ele[1]} : {ele[0]}")
print("Preprocessing data done!")


# <h1>LinearRegressor</h1>

# In[9]:


TrainAndTestModel(LinearRegression(), X_train, y_train, X_test, y_test, "LinearRegressor")


# <h1>OLS</h1>

# In[10]:


X_train_std = np.append(arr=np.ones((len(X_train), 1)).astype(int), values=X_train, axis=1)
print(X_train_std[:5])


# In[11]:


X_opt = X_train_std[:, [0,1,2,3,4,5,6,7,8]]
X_opt = np.array(X_opt, dtype=float)

regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()


# In[12]:


X_opt = X_train_std[:, [0,1,2,3,4,6,7,8]]
X_opt = np.array(X_opt, dtype=float)

regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()


# In[13]:


X_opt = X_train_std[:, [0,2,3,4,6,7,8]]
X_opt = np.array(X_opt, dtype=float)

regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()


# In[14]:


X_opt = X_train_std[:, [0,2,4,6,7,8]]
X_opt = np.array(X_opt, dtype=float)

regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()


# In[15]:


X_opt = X_train_std[:, [0,4,6,7,8]]
X_opt = np.array(X_opt, dtype=float)

regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()


# In[16]:


X_train_opt = X_train[:, [3, 5, 6, 7]]    # select opt's columns
X_test_opt = X_test[:, [3, 5, 6, 7]]   # select opt's columns correspond train


# <h1>Linear opt</h1>

# In[17]:


TrainAndTestModel(LinearRegression(), X_train_opt, y_train, X_test_opt, y_test, "LinearRegressor_opt")


# <h1>SVR</h1>

# In[18]:


TrainAndTestModel(SVR(kernel="linear"), X_train, y_train, X_test, y_test, "SVR")


# <h1>DecisionTreeRegressor</h1>

# In[19]:


TrainAndTestModel(DecisionTreeRegressor(min_samples_split=100), X_train, y_train, X_test, y_test, "DecisionTreeRegressor")


# <h1>RandomForestRegressor</h1>

# In[20]:


TrainAndTestModel(RandomForestRegressor(), X_train, y_train, X_test, y_test, "RandomForestRegressor")


# <h1>RandomForestRegressor OPT</h1>

# In[21]:


TrainAndTestModel(RandomForestRegressor(), X_train_opt, y_train, X_test_opt, y_test, "RandomForestRegressor_opt")


# <h1>Grid Search</h1>

# In[22]:


param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30, 40],'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10, 30], 'max_features': [2, 3, 4, 5, 6]},
]
print(type(param_grid))


# In[23]:


forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X_train, y_train)


# In[24]:


grid_search.best_params_


# In[25]:


grid_search.get_params()


# In[26]:


feature_importances = grid_search.best_estimator_.feature_importances_
sorted(zip(feature_importances, column_name), reverse=True)


# In[27]:


final_model = grid_search.best_estimator_
final_predictions = final_model.predict(X_test)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[28]:


DrawPredict(y_train, final_model.predict(X_train), "GridSearch(RFR)_Train")
DrawPredict(y_test, final_predictions, "GridSearch(RFR)_Test")


# In[29]:


final_rmse


# In[30]:


final_model.score(X_test, y_test)


# <h1>Random search<\h1>

# In[31]:


param_distribs = {
    'n_estimators': randint(low=1, high=300),
    'max_features': randint(low=1, high=9),
}

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=40, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(X_train, y_train)


# In[32]:


rnd_search.best_params_


# In[33]:


feature_importances = rnd_search.best_estimator_.feature_importances_
sorted(zip(feature_importances, column_name), reverse=True)


# In[40]:


# test rmse, mae, acc
rnd_search_model = rnd_search.best_estimator_
rnd_search_predictions = rnd_search_model.predict(X_test)

rnd_search_mse = mean_squared_error(y_test, rnd_search_predictions)
rnd_search_rmse = np.sqrt(rnd_search_mse)
rnd_search_mae = mean_absolute_error(y_test, rnd_search_predictions)
rnd_search_acc = rnd_search_model.score(X_test, y_test)
print(rnd_search_rmse, rnd_search_mae, rnd_search_acc)


# In[44]:


# train rmse, mae
rnd_search_predictions_t = rnd_search_model.predict(X_train)
rnd_search_mse_t = mean_squared_error(y_train, rnd_search_predictions_t)
rnd_search_rmse_t = np.sqrt(rnd_search_mse_t)
rnd_search_mae_t = mean_absolute_error(y_train, rnd_search_predictions_t)
rnd_search_acc_t = rnd_search_model.score(X_train, y_train)

print(rnd_search_rmse_t, rnd_search_mae_t, rnd_search_acc_t)


# In[37]:


DrawPredict(y_train, rnd_search_model.predict(X_train), "RandomSearch(RFR)_Train")
DrawPredict(y_test, rnd_search_predictions, "RandomSearch(RFR)_Test")


# In[38]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
x_poly = poly_reg.fit_transform(X)
regressor_PR = LinearRegression()
regressor_PR.fit(x_poly,y)
acc_PR = regressor_PR.score(x_poly,y)
acc_PR


# In[39]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
TrainAndTestModel(LinearRegression(), poly_reg.fit_transform(X_train), y_train, poly_reg.fit_transform(X_test), y_test, "Polynomial")


# In[ ]:




