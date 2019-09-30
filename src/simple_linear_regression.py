# -*- coding: utf-8 -*-

"""
This is an example to perform simple linear regression algorithm on the dataset (weight and height),
where x = weight and y = height.
"""
import pandas as pd
import numpy as np
import datetime
import random

from sklearn.linear_model import (LinearRegression, 
	Ridge,  
	SGDRegressor)



from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import (SVR, LinearSVR)
from sklearn.metrics import (scorer, r2_score, mean_squared_error, mean_absolute_error)

from sklearn import model_selection
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from utilities.losses import compute_loss
from utilities.optimizers import gradient_descent, pso, mini_batch_gradient_descent
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# General settings
from utilities.visualization import visualize_train, visualize_test


algorithms = {
				'LR': LinearRegression(),
				'KNR': KNeighborsRegressor(),
				'RR':Ridge(),
				'DTR':DecisionTreeRegressor(),
				'RFR': RandomForestRegressor(),
				'GBR': GradientBoostingRegressor(max_depth=10,n_estimators=200),
				'SGDR':SGDRegressor(),
				'SVR':SVR(),
				'MLNR': MLPRegressor(early_stopping=True, learning_rate_init=0.1, momentum=0.2),
				'LSVR': LinearSVR()
		}


seed = 0
# Freeze the random seed
random.seed(seed)
np.random.seed(seed)
train_test_split_test_size = 0.1

# Training settings
alpha = 0.1  # step size
max_iters = 50  # max iterations


def load_data():
	"""
	Load Data from CSV
	:return: df    a panda data frame
	"""
	df = pd.read_csv("../data/diamonds.csv")
	print(df.shape, 'SHAPE')
	print(df.info())
	return df

def convertTo_dummy(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df


def find_outliers(x):
    # detect outliers if below the floor or above the ceiling
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    floor = q1 - 1.5 * iqr
    ceiling = q3 + 1.5 * iqr
    outlier_indices = list(x.index[(x < floor) | (x > ceiling)])
    outlier_values = list(x[outlier_indices])
    return outlier_indices, outlier_values



def data_preprocess(data):
	"""
	Data preprocess:
		1. Split the entire dataset into train and test
		2. Split outputs and inputs
		3. Standardize train and test
		4. Add intercept dummy for computation convenience
	:param data: the given dataset (format: panda DataFrame)
	:return: train_data       train data contains only inputs
			 train_labels     train data contains only labels
			 test_data        test data contains only inputs
			 test_labels      test data contains only labels
			 train_data_full       train data (full) contains both inputs and labels
			 test_data_full       test data (full) contains both inputs and labels
	"""
	# Split the data into train and test
	data = convertTo_dummy(data, ['color'])
	carat_indices, tukey_values = find_outliers(data['carat'])
	data = data.drop(carat_indices)
	depth_indices, tukey_values = find_outliers(data['depth'])
	data = data.drop(depth_indices)


	#import seaborn as sns
	#import matplotlib.pyplot as plt
	#sns.boxplot(x=data[''])
	#plt.show()

	train_data, test_data = train_test_split(data, test_size = train_test_split_test_size)

	# Pre-process data (both train and test)
	train_data_full = train_data.copy()
	train_data = train_data.drop(["price"], axis = 1)
	train_data = train_data.drop(["index"], axis = 1)
	train_labels = train_data_full["price"]

	test_data_full = test_data.copy()
	test_data = test_data.drop(["price"], axis = 1)
	test_data = test_data.drop(["index"], axis = 1)
	test_labels = test_data_full["price"]

	le = LabelEncoder()
	train_data.cut = le.fit_transform(train_data.cut) 				# return orginal labels for categorical
	test_data.cut = le.fit_transform(test_data.cut)
	train_data.clarity = le.fit_transform(train_data.clarity)
	test_data.clarity = le.fit_transform(test_data.clarity)




	# Standardize the inputs
	train_mean = train_data.mean()
	train_std = train_data.std()
	train_data = (train_data - train_mean) / train_std
	test_data = (test_data - train_mean) / train_std




	# Tricks: add dummy intercept to both train and test
   # train_data['intercept_dummy'] = pd.Series(1.0, index = train_data.index)
	#test_data['intercept_dummy'] = pd.Series(1.0, index = test_data.index)
	return train_data, train_labels, test_data, test_labels, train_data_full, test_data_full


	
def buildModel(train_data, train_labels, algorithm):

	start_time = datetime.datetime.now()



	trainedModel = algorithm.fit(train_data, train_labels)
	end_time = datetime.datetime.now()  # Track learning ending time
	exection_time = (end_time - start_time).total_seconds()  # Track execution time

	return trainedModel, exection_time


def predict_testModel(test_data, test_labels, model):

	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	pred = model.predict(test_data)
	mse = mean_squared_error(test_labels, pred)
	print('MSE+ {error}'.format(error=mse))
	print('RSME+{error}'.format(error=np.sqrt(mse)))
	r2_error = r2_score(test_labels, pred)
	print('R2: {error}'.format(error=r2_error))
	# adjusted r squared where p is the number of features
	adjusted_r2_error = 1-(1-r2_error)*(len(test_data.index)-1)/(len(test_data.index)-len(test_data.columns)-1)
	print('ADJUSTED R2: {error}'.format(error=adjusted_r2_error))
	mae = mean_absolute_error(test_labels, pred)
	print('Mae absolute {error}'.format(error=mae))



if __name__ == '__main__':


	# Settings
	#metric_type = "MSE"  # MSE, RMSE, MAE, R2
	#optimizer_type = "BGD"  # PSO, BGD

	# Step 1: Load Data
	data = load_data()

	# Step 2: Preprocess the data
	
	train_data, train_labels, test_data, test_labels, train_data_full, test_data_full = data_preprocess(data)


	print(train_data.head(5))

	for key, value in algorithms.items():
		print('========================================')
		print('Executing Model '+key)
		model, execution_time = buildModel(train_data, train_labels, value)
		print("Learn: execution time= "+str(execution_time))
		predict_testModel(test_data, test_labels, model)




	# Build baseline model
   # print("R2:", -compute_loss(test_labels.values, test_data.values, thetas[-1], "R2"))  # R2 should be maximize
	#print("MSE:", compute_loss(test_labels.values, test_data.values, thetas[-1], "MSE"))
	#print("RMSE:", compute_loss(test_labels.values, test_data.values, thetas[-1], "RMSE"))
	#print("MAE:", compute_loss(test_labels.values, test_data.values, thetas[-1], "MAE"))

