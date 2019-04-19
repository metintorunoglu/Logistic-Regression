# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import the dataset
dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:, [2,3]].values
y=dataset.iloc[:, -1].values

#Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#import Logistic Regression and fit the model
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#Predicting the test results
y_pred=classifier.predict(X_test)

#Making the Confusion Matrix(type "cm" on the python shell after execution)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
classifier.score(X_test, y_test)










