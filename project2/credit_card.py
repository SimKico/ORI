#Basic python library which need to import
import inline as inline
import pandas as pd
import numpy as np


import seaborn as sns

#Date stuff
from datetime import datetime
from datetime import timedelta

#Library for Nice graphing
#import seaborn as sns
import matplotlib.pyplot as plt
#import statsmodels.formula.api as sn


#Library for statistics operation
import scipy.stats as stats

# Date Time library
from datetime import datetime

#Machine learning Library
#import statsmodels.api as sm
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import warnings
warnings.filterwarnings('ignore')

#Kmeans
#from kmeans import KMeans

train_data = pd.read_csv('data/credit_card_data.csv')
print("Data Types:", train_data.dtypes)

print(' {} - rows \n {} - columns.\n'.format(train_data.shape[0],train_data.shape[1]))

# Find the total number of missing values in the dataframe
print ("\nMissing values :  ", train_data.isnull().sum().values.sum())

# printing total numbers of Unique value in the dataframe.
print ("\nUnique values :  \n", train_data.nunique())

# print data to see which one are missing
print ("\nMissing values :  \n", train_data.isnull().any())

# CREDIT_LIMIT  and MINIMUM_PAYMENTS has missing values so we need to remove with median
train_data['CREDIT_LIMIT'].fillna(train_data['CREDIT_LIMIT'].median())
train_data['CREDIT_LIMIT'].count()
train_data['MINIMUM_PAYMENTS'].median()
train_data['MINIMUM_PAYMENTS'].fillna(train_data['MINIMUM_PAYMENTS'].median())

# checking the replacement
print("\nMissing values :  \n", train_data.isnull().any())

train_clean = train_data.apply(lambda x:x.fillna(x.value_counts().index[0]))
train_clean = train_clean.drop(['CUST_ID'], axis=1)

# StandardScaler performs the task of Standardization. Usually a dataset contains variables that are different in scale.
scaler = StandardScaler()
train_data_scl= scaler.fit_transform(train_clean.values)
train_clean = pd.DataFrame(train_data_scl,columns=train_clean.columns)

#train_clean = pd.DataFrame(x_scaled,columns=train_clean.columns)

# Normalization refers to rescaling real valued numeric attributes into the range 0 and 1.
# It is useful to scale the input attributes for a model that relies on the magnitude of values, such as distance measures used in unsupervised learning.
norm = normalize(train_data_scl)

# We can apply both (StandartScaler and Normalize) on our data before clustering.
df_norm=pd.DataFrame(norm)



#Kmeans
Sum_of_squared_distances = []
K = range(1, 15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(train_clean)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
x_pos = 1.5
y_pos = 140000
plt.text(x_pos, y_pos, 'After 8 clusters adding more gives minimal benefit to the model.')
plt.show()

km = KMeans(n_clusters=8)
km = km.fit_predict(train_clean)

# 3D data
pca = PCA(n_components=3).fit_transform(train_clean)
fig = plt.figure(figsize=(8, 7), dpi=80, facecolor='w', edgecolor='k')
ax = plt.axes(projection="3d")
ax.scatter3D(pca.T[0],pca.T[1],pca.T[2],c=km,cmap='Spectral')
xLabel = ax.set_xlabel('X')
yLabel = ax.set_ylabel('Y')
zLabel = ax.set_ylabel('Z')
plt.show()
print(Sum_of_squared_distances)

# 2D data
pca2 = PCA(n_components=2).fit_transform(train_clean)
plt.scatter(pca2[:, 0], pca2[:, 1], c=km, s=50, cmap='viridis')
plt.show()

np.random.seed(0)
msk = np.random.rand(len(train_clean)) < 0.8
trained_data = train_clean[msk]
test_data = train_clean[~msk]

data = np.array(trained_data)
data_test = np.array(test_data)

kmeans = KMeans(n_clusters=8, random_state=0).fit(data)

y_k = kmeans.predict(data_test)
test_data['PREDICTED_CLUSTER'] = y_k
train_summary = test_data.groupby(by='PREDICTED_CLUSTER').mean()
train_summary = train_summary[['BALANCE', 'PURCHASES',
                               'PURCHASES_FREQUENCY','CREDIT_LIMIT',
                               'ONEOFF_PURCHASES_FREQUENCY',
                              'MINIMUM_PAYMENTS','PRC_FULL_PAYMENT',
                               'PAYMENTS']]
print(train_summary)


plt.figure(figsize=(15,15),dpi=60, facecolor='w', edgecolor='k')
for i, j in enumerate(train_summary):
    plt.subplot(3, 3, i+1)

    sns.barplot(train_summary.index, train_summary[j])
    plt.title(j, fontdict={'color': 'darkblue'})

plt.tight_layout()
plt.show()



