# California-Housing-Price-Prediction
The purpose of the project is to predict median house values in Californian districts, given many features from these districts.

Build a model to predict house price using 'California Housing' dataset.
Load the dataset using from sklearn. datasets import fetch_california_housing and california_housing = fetch_california_housing. Build a model of housing prices to predict median house values in California using the provided dataset. Split the dataset into training and test sets using random_state = 2023, and 90% in training set. Show the EDA and preprocessing steps Train the model to learn from the data to predict the median housing price in any district, given all the other metrics. Predict median house values in Californian districts, given many features from these districts, compare the predicted values with actrel Values, reluate your model performance using different metrics for the test data. Are You satisfied with the model performance? Why or why not? Apply the techniques for improving the performance of your model. Perform hyperparameter tuning and find your model with optimum hyperparameter aluesompare the performance of your initial model with optimized model Can hyperparamster tuning always give you improved model?

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# Load the dataset
california_housing = fetch_california_housing(as_frame=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    california_housing.data, california_housing.target, test_size=0.1, random_state=2023)
Now let's perform some EDA and preprocessing steps on the data. We can start by examining the shape of the training and test sets:
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

california_housing.data.head()
Training set shape: (18576, 8)
Test set shape: (2064, 8)
MedInc	HouseAge	AveRooms	AveBedrms	Population	AveOccup	Latitude	Longitude
0	8.3252	41.0	6.984127	1.023810	322.0	2.555556	37.88	-122.23
1	8.3014	21.0	6.238137	0.971880	2401.0	2.109842	37.86	-122.22
2	7.2574	52.0	8.288136	1.073446	496.0	2.802260	37.85	-122.24
3	5.6431	52.0	5.817352	1.073059	558.0	2.547945	37.85	-122.25
4	3.8462	52.0	6.281853	1.081081	565.0	2.181467	37.85	-122.25
print("Missing values in training set:", np.isnan(X_train).sum())
print("Missing values in test set:", np.isnan(X_test).sum())
Missing values in training set: 0
Missing values in test set: 0
Since there are no missing values, we can proceed to scaling the data using the StandardScaler:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
Now we can train a linear regression model on the training data:
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
LinearRegression()
We can use the trained model to predict median house values in Californian districts:
y_pred = model.predict(X_test)
To evaluate the performance of the model, we can use various metrics such as the mean squared error (MSE) and the coefficient of determination (R-squared):
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R-squared:", r2)
MSE: 4994.191354320657
R-squared: -3775.001361014611
If we are not satisfied with the performance of the model, we can try various techniques for improving its performance such as feature engineering, using a different model or hyperparameter tuning.

For hyperparameter tuning, we can use GridSearchCV to search over a range of hyperparameters and find the model with the optimum hyperparameter values:

from sklearn.model_selection import GridSearchCV

params = {'normalize': [True, False], 'fit_intercept': [True, False]}
grid_search = GridSearchCV(LinearRegression(), params, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
Best parameters: {'fit_intercept': True, 'normalize': True}
Best score: 0.599216843895012
We can compare the performance of the initial model with the optimized model using the same metrics:
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Initial model MSE:", mse)
print("Initial model R-squared:", r2)

model = LinearRegression(normalize=True, fit_intercept=False)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Optimized model MSE:", mse)
print("Optimized model R-squared:", r2)
Initial model MSE: 0.5074732599633678
Initial model R-squared: 0.6163103124508021
Optimized model MSE: 0.5986141371568142
Optimized model R-squared: 0.5474006428145387
