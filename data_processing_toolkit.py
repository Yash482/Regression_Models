#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#getting dataset
dataset = pd.read_csv('Data.csv')
X= dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#taking care of missing data
"""
we must apply this when any data is missing
it is also suggested to apply this when working on large amount of data
"""
from sklearn.impute import SimpleImputer
#np.nan corresponds to index of missing value
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
#.fit method takes 2 para
#1st no. of rows i.e. all here
#2nd index of columns 
imputer.fit(X[:, :-1])

X[:, :-1] = imputer.transform(X[:, :-1])
#no missing data now

#encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
index_of_col = 0
ct = ColumnTransformer(transformers = ['encoder' , OneHotEncoder(), [index_of_col]], remainder = 'passthrough')
X= np.array(ct.fit_transform(X))

#splitting data for test and training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, :-1] = sc.fit_transform(X_train[:, :-1])
X_test[:, :-1] = sc.transform(X_test[:, :-1])