#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#getting dataset
dataset = pd.read_csv('Data.csv')
X= dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#splitting data for test and training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train )

#now our model is ready

#predicting result
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#model performance r^2
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))