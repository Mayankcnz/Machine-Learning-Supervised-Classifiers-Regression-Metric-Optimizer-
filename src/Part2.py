import pandas as pd
import numpy as np
import time
import datetime
import random
from sklearn.preprocessing import Imputer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import sklearn.feature_selection
from sklearn.feature_extraction import DictVectorizer
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

seed = 0
# Freeze the random seed
random.seed(seed)
np.random.seed(seed)
train_test_split_test_size = 0.1

models = {
    'LR': LogisticRegression(),
    'LDA': LinearDiscriminantAnalysis(),
    'KNN': KNeighborsClassifier(),
    'CART': DecisionTreeClassifier(),
    'NB': GaussianNB(),
    'RFC': RandomForestClassifier(),
    'ABC': AdaBoostClassifier(),
    'SVM': SVC(),
    'GBC': GradientBoostingClassifier(),
    'MLP': MLPClassifier()
}


def load_data():
    df = pd.read_csv("../data/adult.csv")
    # test_df = pd.read_csv('../data/adult.test', skiprows=[0], sep=',', header=None)
    d = {' <=50K': 0, ' >50K': 1}
    df['Income'] = df['Income'].map(d).fillna(df['Income'])

    tukey_indices, tukey_values = find_outliers(df['Age'])  # find outliers use IQR method
    # print('Outliers ')
    # print(len(tukey_indices))
    df = df.drop(tukey_indices, axis=0)
    # print('rip')

    X = df.drop(['Income'], axis=1)
    y = df.Income

    cols = ['workclass', 'occupation', 'native-country']
    # df['occupation'] = df['occupation'].replace(to_replace=' ?',value=np.nan)
    X = X.replace(' ?', value=np.NaN)
    # now impute
    y = df.Income.replace(' ?', value=np.nan)

    print(X.info)
    sns.heatmap(X.isnull(), cbar=False)
    plt.show()
    X[cols] = X[cols].fillna(X.mode().iloc[0])
    cat_df_income = X.select_dtypes(include=['object'].copy())




    cat_df_income_copy = cat_df_income.copy()

    #for col in cols:
     #   X[col] = cat_df_income_copy[col].astype('category')

    X['native-country'] = ['United-States' if x == ' United-States' else 'Other' for x in
                           X['native-country']]  # combined the category levels

    carrier_count = y.value_counts()
    sns.set(style="darkgrid")
    sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
    plt.title('Frequency Distribution of native-country')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Income 0: <=50k 1: >50k', fontsize=12)
    plt.show()
    plt.show(block=True)
    plt.interactive(False)
    # apply categorical encoding and one hot encoding
    labelencoder_X = LabelEncoder()
    todummy_list = ['marital-status', 'Farming-fishing', 'race', 'sex','native-country']

    for col in cat_df_income.columns:
        if col not in todummy_list:
            X[col] = labelencoder_X.fit_transform(X[col])


    # handle ordinal features that does not have a relationship
    X = convertTo_dummy(X, todummy_list)

    # fill missing values in the income column using median approach instead of mode
    y.income = df['Income'].fillna(X.mode().iloc[0])

    # =================== outlier detection ============================

    # ========================== Distribution of Features =======================

    # visualize_Features(df['Age'])
    # plot_histogram_dv(df['Age'], trin_labels)

    # add interactions

    X = add_interactions(X) # add possible interactions between features

    # Normalize total_bedrooms column
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(x_scaled)

    # dimensionality reduction using PCA

    # one downside is a bit hard to interpret
    #X_pca = PCA(n_components=10)  # its foe soime reason reducing sopatial infromaiton and giving low accuracy so i decided to not use it

    #X_1 = pd.DataFrame(X_pca.fit_transform(X))

    return X, y  # return data and labels


def convertTo_dummy(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df


# can also use standard deviation to find outliers but iqr is better because does not make any assumptions on the normality of the data
# other way could be kernal density which is non paramtetic way to estimate probability density function of a given feature, has the ability to capture bimodal distributions
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


# histograms
def visualize_Features(x):
    plt.hist(x, color='gray', alpha=0.5)
    plt.title("Histogram of '{var name}'".format(var_name=x.name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


def plot_histogram_dv(x, y):
    plt.hist(list(x[y == 0]), alpha=0.5, label='DV=0')
    plt.hist(list(x[y == 1]), alpha=0.5, label='DV=1')
    plt.title("Histogram of '{var name}'".format(var_name=x.name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.show()


from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures


def add_interactions(df):
    # Get feature names
    combos = list(combinations(list(df.columns), 2))
    colnames = list(df.columns) + ['_'.join(x) for x in combos]

    # find interactions
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = colnames

    # remove interaction terms with 0 value
    noint_indicies = [i for i, x in enumerate(list((df == 0).all())) if x]
    df = df.drop(df.columns[noint_indicies], axis=1)
    return df

    '''
    workclass = oridinal
    education = ordinal
    marital-status = nominal
    occupation = ordinal
    Farming-fishing = nominal
    race = nominal
    sex = nominal
    native-country = nominal
    '''

    # turn each row as key-value pairs


# cat_df_income_onehotDict = cat_df_income_onehot.to_dict(orient='records')
#  print(cat_df_income_onehotDict)
# dv_X = DictVectorizer(sparse=False) # not a sparse matrix
# x_encoded = dv_X.fit_transform(cat_df_income_onehotDict)
#    cat_df_income_onehot = pd.get_dummies(cat_df_income_onehot, columns=['occupation'], prefix=['occupation'])



def predictTest(x_test, model):

    y_pred = model.predict(x_test)
    return y_pred


from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, precision_score, recall_score)

def calculateScore(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy :", accuracy)
    prescision = precision_score(y_test, y_pred)
    print("prescision:", prescision)
    recall = recall_score(y_test, y_pred)
    print("recall:", recall)
    f1 = f1_score(y_test, y_pred)
    print("f1:", f1)
    rauc = roc_auc_score(y_test, y_pred)
    print("rauc:", rauc)


def processData(X, y):
    # X = X.iloc[1:] # delete the header , as not sure if the machine learning algorithm is capable of dealing with it

    print("===================== Processing Data ======================================")

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=1)


    print()


    '''
    select = sklearn.feature_selection.SelectKBest(k=20)
    selected_features = select.fit(x_train, y_train)
    indices_selected = selected_features.get_support(indices=True)
    colnames_selected = [X.columns[i] for i in indices_selected]

    x_train_selected = x_train[colnames_selected]
    x_test_selected = x_test[colnames_selected]
    '''

    for model_type, model in models.items():
        print("Running model %s",model_type);
        print(' ========================================= ')
        my_model = model.fit(x_train, y_train)
        predicted_model = predictTest(x_test, my_model)
        calculateScore(predicted_model, y_test)

    # impute values
    # do hot encoding
    # dummy values trap

    # classifer

    # imputer = Imputer(missing_values==)
    '''
    for name,models in model:
        kfold = model_selection.KFold(n_splits=10, random_state=7)
        cv_result = model_selection.cross_val_score(models,train_data, train_labels, cv=kfold, scoring='accuracy')
        result.append(cv_result)
        names.append(name)
        msg = "%s,%f(%f)"%(name, cv_result.mean(), cv_result.std())
        print(msg)
    '''

    return x_test, y_test, x_train, y_train, model



if __name__ == '__main__':
    X, y = load_data()
    x_test, y_test, x_train, y_train, model = processData(X, y)
