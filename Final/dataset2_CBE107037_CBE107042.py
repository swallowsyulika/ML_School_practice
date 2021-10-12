#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from scipy.stats import randint
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
import seaborn as sns


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
            print(f"column {index} have missing data, fixed!")
        else:
            print(f"column {index} not have missing data")
            
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
    plt.xlabel("dataNums")
    plt.ylabel("charges")
    plt.title(title)
    plt.show() 


# In[7]:


def TrainAndTestModel(model, X_train, y_train, X_test, y_test, title, want_fit_model=False):
    if want_fit_model:
        model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    train_acc = model.score(X_train, y_train)
    #train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    #DrawPredict(y_train, train_pred, title+"_Train" + "\n" + f"ACC: {train_acc}")
    
    test_pred = model.predict(X_test)
    test_acc = model.score(X_test, y_test)
    #test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    #DrawPredict(y_test, test_pred, title+"_Test" + "\n" + f"ACC: {test_acc}")
    
    train_cm = confusion_matrix(y_train, train_pred)
    test_cm = confusion_matrix(y_test, test_pred)
    print('train_cm: ','\n',train_cm,'\n')
    print('test_cm: ','\n',test_cm,'\n')
    print('train_acc: ',train_acc)
    print('test_acc: ',test_acc)
    train_cm_matrix = pd.DataFrame(data=train_cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])
    test_cm_matrix = pd.DataFrame(data=test_cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])
    
    
    
    return model,train_cm_matrix,test_cm_matrix


# In[8]:


def show_cm(train_cm='',test_cm=''):
        sns.heatmap(train_cm, annot=True, fmt='d', cmap='viridis')


# In[9]:


df = pd.read_csv("./dataset/final_project_dataset_2.csv")
df.head(5)


# In[10]:


dfX = df.iloc[:, 1:-1]
X = dfX.values
dfy = df.iloc[:, [-1]]
y = dfy.values
X_column_name = np.array(dfX.columns.values)
y_column_name = np.array(dfy.columns.values)

X = DealMissingData(X, dfX)
y = DealMissingData(y, dfy)


print(X[0])
print(y[0])

X,X_column_name = MakeOneHot(X,X_column_name)
y,y_column_name = MakeOneHot(y,y_column_name)
y = y.reshape(1, -1)[0]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train, X_test = NormalizeData(X_train, X_test)
print("----------- X column -----------")
for ele in zip(X[0], X_column_name):
    print(f"{ele[1]} : {ele[0]}")
print("----------- y column -----------")
for ele in zip(y, y_column_name):
    print(f"{ele[1]} : {ele[0]}")
print("Preprocessing data done!")


# <h1>LogisticRegression<\h1>

# In[11]:


Logistic_model, Log_train_cm, Log_test_cm = TrainAndTestModel(LogisticRegression(random_state=42,verbose=1), X_train, y_train, X_test, y_test, "Logistic_classifier",want_fit_model=True)


# In[52]:


sns.heatmap(Log_train_cm, annot=True, fmt='d', cmap='Blues')
print('Log_train')


# In[53]:


sns.heatmap(Log_train_cm, annot=True, fmt='d', cmap='Blues')
print('Log_test')


# <h1>DecisionTree with Randomsearch<\h1>

# In[22]:


param_distribs = {
    'min_samples_split': randint(low=100, high=200),
    'max_features': randint(low=70, high=95),
    'max_depth': randint(low=3, high=20),
    'min_samples_leaf': randint(low=40, high=50),
}


# In[16]:


DecisionTree = DecisionTreeClassifier(criterion='entropy', random_state=0)
rnd_search = RandomizedSearchCV(DecisionTree, param_distributions=param_distribs,
                                n_iter=50, cv=3, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(X_train, y_train)


# In[26]:


rnd_search.best_params_


# In[27]:


feature_importances = rnd_search.best_estimator_.feature_importances_
sorted(zip(feature_importances, X_column_name), reverse=True)


# In[28]:


Decision_rnd_search_model = rnd_search.best_estimator_
Decision_model, Decision_train_cm, Decision_test_cm = TrainAndTestModel(Decision_rnd_search_model, X_train, y_train, X_test, y_test, "DecisionTree_classifier")


# In[50]:


sns.heatmap(Decision_train_cm, annot=True, fmt='d', cmap='Blues')
print('Decision_train')


# In[51]:


sns.heatmap(Decision_test_cm, annot=True, fmt='d', cmap='Blues')
print('Decision_test')


# <h1>RandomForest with Random Search<\h1>

# In[19]:


param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=107),
    'max_depth': randint(low=3, high=50),
    'min_samples_leaf': randint(low=5, high=50),
}

forest_classifier = RandomForestClassifier(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=100, cv=3, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(X_train, y_train)


# In[32]:


rnd_search.best_params_


# In[33]:


feature_importances = rnd_search.best_estimator_.feature_importances_
sorted(zip(feature_importances, X_column_name), reverse=True)


# In[34]:


rnd_search_model = rnd_search.best_estimator_


# In[36]:


RandomForest_model, RandomForest_train_cm, RandomForest_test_cm = TrainAndTestModel(rnd_search_model, X_train, y_train, X_test, y_test, "RandomForest_classifier")


# In[54]:


sns.heatmap(RandomForest_train_cm, annot=True, fmt='d', cmap='Blues')
print('RandomForest_train')


# In[55]:


sns.heatmap(RandomForest_test_cm, annot=True, fmt='d', cmap='Blues')
print('RandomForest_test')


# <h1>SVC<br>!!!NEED A LOT OF TIME!!!<\h1> 

# In[56]:


from datetime import datetime
now = datetime.now()
print("now =", now)


# In[57]:


SVC_model,SVC_train_cm,SVC_test_cm = TrainAndTestModel(SVC(verbose=True), X_train, y_train, X_test, y_test, "SVC",want_fit_model=True)


# In[58]:


now = datetime.now()
print("now =", now)


# In[59]:


sns.heatmap(SVC_train_cm, annot=True, fmt='d', cmap='Blues')
print('SVC_train')


# In[60]:


sns.heatmap(SVC_test_cm, annot=True, fmt='d', cmap='Blues')
print('SVC_test')


# In[ ]:




