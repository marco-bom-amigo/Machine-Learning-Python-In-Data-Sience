# Data Preprocessing

# Importig the librares
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv("data/Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
