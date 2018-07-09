# Data Preprocessing

# Importig the librares
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("data/Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Taking care of missing data
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
