#Basic python library which need to import
import inline as inline
import pandas as pd
import numpy as np

#Date stuff
from datetime import datetime
from datetime import timedelta

#Library for Nice graphing
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as sn


#Library for statistics operation
import scipy.stats as stats

# Date Time library
from datetime import datetime

#Machine learning Library
import statsmodels.api as sm
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import warnings
warnings.filterwarnings('ignore')


train_data = pd.read_csv('data\data.csv')
print("Data Types:", train_data.dtypes)

print(' {} - rows \n {} - columns.\n'.format(train_data.shape[0],train_data.shape[1]))

# Find the total number of missing values in the dataframe
print ("\nMissing values :  ", train_data.isnull().sum().values.sum())

# printing total numbers of Unique value in the dataframe.
print ("\nUnique values :  \n",train_data.nunique())

# print data to see which one are missing
print ("\nMissing values :  \n",train_data.isnull().any())

# CREDIT_LIMIT  and MINIMUM_PAYMENTS has missing values so we need to remove with median
train_data['CREDIT_LIMIT'].fillna(train_data['CREDIT_LIMIT'].median(),inplace=True)
train_data['CREDIT_LIMIT'].count()
train_data['MINIMUM_PAYMENTS'].median()
train_data['MINIMUM_PAYMENTS'].fillna(train_data['MINIMUM_PAYMENTS'].median(),inplace=True)

# checking the replacement
print ("\nMissing values :  \n",train_data.isnull().any())