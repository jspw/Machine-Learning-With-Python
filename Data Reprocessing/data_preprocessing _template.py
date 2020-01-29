# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 05:30:46 2020

@author: JackSparrow
"""

# importing libraries

import numpy as np  #for mathematics
import matplotlib.pyplot as plt #for plot
import pandas as pd #to import dataset and manage dataset


#importing dataset

dataset = pd.read_csv("Data.csv")

X = dataset.iloc[:, :-1].values #take all collum except the last one thats wht -1
Y = dataset.iloc[:,3].values #take the 3no colum/purchase colum (index start with 0)

"""
#taking care of missing data
#from sklearn.preprocessing import Imputer #this shit is not working in dnt know why
from sklearn.impute import SimpleImputer 

imputer = SimpleImputer(missing_values = np.nan,strategy = "mean",verbose=0) #missing datas are filled as 'nan'
imputer = imputer.fit(X[:,1:3]) # as we need to imput the datas of 1,2 ,python has a border
X[:,1:3] = imputer.transform(X[:,1:3])


#encoding the catagorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer 

labelencoder_X  = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0]) #we need to encode the name of the country as they are in strings
onehotencoder = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
X = onehotencoder.fit_transform(X)


#encode Y
labelencoder_Y  = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

"""


#spliting data for training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test  = train_test_split(X,Y,test_size=0.2,random_state = 0)

"""

# scalling datas in a fixed range
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

"""





















