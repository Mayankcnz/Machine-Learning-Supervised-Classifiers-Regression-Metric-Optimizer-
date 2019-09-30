# -*- coding: utf-8 -*-

"""
This is an example to perform simple linear regression algorithm on the dataset (weight and height),
where x = weight and y = height.
"""
import pandas as pd
import numpy as np
import datetime
import random

from sklearn.linear_model import (LinearRegression)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (scorer, r2_score, mean_squared_error, mean_absolute_error)

from sklearn import model_selection
from utilities.losses import compute_loss
from utilities.optimizers import gradient_descent, pso, mini_batch_gradient_descent
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utilities.visualization import visualize_train, visualize_test
# General settings
from utilities.visualization import visualize_train, visualize_test, predict


seed = 0
# Freeze the random seed
random.seed(seed)
np.random.seed(seed)
train_test_split_test_size = 0.1

# Training settings
alpha = 0.3  # step size
max_iters = 50  # max iterations


def load_data():
    #Load Data from CSV

    #df = pd.read_csv("../data/Part2.csv")
    df=pd.read_csv("../data/Part2Outliers.csv")
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

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.boxplot(x=data[''])
    # plt.show()

    # Split the data into train and test
    train_data, test_data = train_test_split(data, test_size=train_test_split_test_size)

    # Pre-process data (both train and test)
    train_data_full = train_data.copy()
    train_data = train_data.drop(["Height"], axis=1)
    train_labels = train_data_full["Height"]

    test_data_full = test_data.copy()
    test_data = test_data.drop(["Height"], axis=1)
    test_labels = test_data_full["Height"]

    # Standardize the inputs
    train_mean = train_data.mean()
    train_std = train_data.std()
    train_data = (train_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    # Tricks: add dummy intercept to both train and test
    train_data['intercept_dummy'] = pd.Series(1.0, index = train_data.index)
    test_data['intercept_dummy'] = pd.Series(1.0, index = test_data.index)
    return train_data, train_labels, test_data, test_labels, train_data_full, test_data_full


def learn(y, x, theta, max_iters, alpha, optimizer_type, metric_type):
    """
    Learn to estimate the regression parameters (i.e., w and b)
    :param y:                   train labels
    :param x:                   train data
    :param theta:               model parameter
    :param max_iters:           max training iterations
    :param alpha:               step size
    :param optimizer_type:      optimizer type (default: Batch Gradient Descient): GD, SGD, MiniBGD or PSO
    :param metric_type:         metric type (MSE, RMSE, R2, MAE). NOTE: MAE can't be optimized by GD methods.
    :return: thetas              all updated model parameters tracked during the learning course
             losses             all losses tracked during the learning course
    """

    print('---------------------------------------')
    thetas = None
    losses = None
    if optimizer_type == "BGD":
        thetas, losses = gradient_descent(y, x, theta, max_iters, alpha, metric_type)
        print('called')
    elif optimizer_type == "MiniBGD":
        thetas, losses = mini_batch_gradient_descent(y, x, theta, max_iters, alpha, metric_type, mini_batch_size = 10)
    elif optimizer_type == "PSO":
        thetas, losses = pso(y, x, theta, max_iters, 100, metric_type)
    else:
        raise ValueError(
            "[ERROR] The optimizer '{ot}' is not defined, please double check and re-run your program.".format(
                ot = optimizer_type))
    return thetas, losses



def buildModel(train_data, train_labels, algorithm):
    start_time = datetime.datetime.now()

    trainedModel = algorithm.fit(train_data, train_labels)
    end_time = datetime.datetime.now()  # Track learning ending time
    exection_time = (end_time - start_time).total_seconds()  # Track execution time

    return trainedModel, exection_time


def predict_testModel(pred, test_labels):
    mse = mean_squared_error(test_labels, pred)
    print('MSE+ {error}'.format(error=mse))
    print('RSME+{error}'.format(error=np.sqrt(mse)))
    r2_error = r2_score(test_labels, pred)
    print('R2: {error}'.format(error=r2_error))
    # adjusted r squared where p is the number of features
    adjusted_r2_error = 1 - (1 - r2_error) * (len(test_data.index) - 1) / (
                len(test_data.index) - len(test_data.columns) - 1)
    print('ADJUSTED R2: {error}'.format(error=adjusted_r2_error))
    mae = mean_absolute_error(test_labels, pred)
    print('Mae absolute {error}'.format(error=mae))


if __name__ == '__main__':

    print('here')
    # Settings
    metric_type = "MSE"  # MSE, RMSE, MAE, R2
    optimizer_type = "PSO"  # PSO, BGD

    # Step 1: Load Data
    data = load_data()

    # Step 2: Preprocess the data

    train_data, train_labels, test_data, test_labels, train_data_full, test_data_full = data_preprocess(data)

    theta = np.array([0.0, 0.0])  # Initialize model parameter

    start_time = datetime.datetime.now()  # Track learning starting time
    print('Going In')
    thetas, losses = learn(train_labels.values, train_data.values, theta, max_iters, alpha, optimizer_type, metric_type)

    print('===================== HERE')

    end_time = datetime.datetime.now()  # Track learning ending time
    exection_time = (end_time - start_time).total_seconds()  # Track execution time

    # Step 4: Results presentation
    print("Learn: execution time={t:.3f} seconds".format(t=exection_time))

    import matplotlib.pyplot as plt


    # Build baseline model
    print("R2:", -compute_loss(test_labels.values, test_data.values, thetas[-1], "R2"))  # R2 should be maximize
    print("MSE:", compute_loss(test_labels.values, test_data.values, thetas[-1], "MSE"))
    print("RMSE:", compute_loss(test_labels.values, test_data.values, thetas[-1], "RMSE"))
    print("MAE:", compute_loss(test_labels.values, test_data.values, thetas[-1], "MAE"))





    y_pred = predict(test_data, thetas[-1])


    print('printing predicted =================')
    #print(y_pred)

    predict_testModel(y_pred, test_labels)


    visualize_test(test_data_full, test_data, thetas)
    #visualize_train(train_data_full, test_labels, train_data, thetas, losses, 50)
    plt.show()
    plt.title('Cost Function J')
    plt.xlabel('No. of iterations')
    plt.ylabel('Cost')
    plt.plot(losses)
    plt.show()
