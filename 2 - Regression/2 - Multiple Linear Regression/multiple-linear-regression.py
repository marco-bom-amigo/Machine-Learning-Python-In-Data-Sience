# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

# Importing the dataset
dataset = pd.read_csv("data/50_Startups.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Encoding categorical data
labelenconder_X = LabelEncoder()
X[:,3] = labelenconder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:,1:]

# Splitting the dataset into the Training set and the Test set
X_train, X_test,Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the training data
regressor = LinearRegression(fit_intercept=True, n_jobs=-1, normalize=False)
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
print(regressor_OLS.summary())