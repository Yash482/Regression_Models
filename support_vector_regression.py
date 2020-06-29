#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#getting dataset
dataset = pd.read_csv('Data.csv')
X= dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#reshape y
y = y.reshape(len(y) , 1)

#splitting data for test and training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature scaling is compulsory here
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_train = sc_x.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

#training the model
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train )

#now our model is ready

#predicting result
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(X_test)))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#model performance r^2
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))