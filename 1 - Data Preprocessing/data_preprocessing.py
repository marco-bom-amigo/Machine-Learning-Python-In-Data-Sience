# Data Preprocessing

# Importig the librares
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("data/Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Taking care of missing data
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# Encoding categorical data
labelenconder_X = LabelEncoder()
X[:,0] = labelenconder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelenconder_Y = LabelEncoder()
Y = labelenconder_Y.fit_transform(Y)