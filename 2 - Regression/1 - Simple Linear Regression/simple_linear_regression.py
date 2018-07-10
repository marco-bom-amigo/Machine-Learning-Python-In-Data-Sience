# Simple Linear Regression

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv("data/Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and the Test set
X_train, X_test,Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# Fitting Simple Linear Regression to the training data
regressor = LinearRegression(fit_intercept=True, n_jobs=-1)
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, Y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary x Experience (Training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

# Visualising the Test set results
plt.scatter(X_test, Y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary x Experience (Test set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()