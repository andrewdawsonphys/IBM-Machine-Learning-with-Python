# IMPORT packages required
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline

# The data set which will be used is the FuelConsumptionCo2, which was used in the two previous labs

"""
Sometimes the data is not really linear and looks curvy. In this case we use polynomical regression methods.
Many regression models exist that can be used to fit whatever the dataset looks like e.g quadratic, cubic, quartic....


Polynomial regression can be used where the relationship between the independent variable x and the dependent variable y is modelled
as an n'th degree polynomial (See notes for further discussion)

"""

# Reading in the data
df = pd.read_csv("FuelConsumptionCo2.csv")

df.head()

# Plotting emission values to engine size
plt.scatter(df.ENGINESIZE,df.CO2EMISSIONS)
plt.xlabel('Engine Size')
plt.ylabel('Emission')

#Select some features
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

# Creating train and test data
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test= cdf[~msk]

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train['CO2EMISSIONS'])

test_x = np.asanyarray(test['ENGINESIZE'])
test_y = np.asanyarray(test['CO2EMISSIONS'])

# Polynomial features will generate a matrix consisting of all polynomial combinations of the features with degree
# less than or equal to the specified degree.
poly = PolynomialFeatures(degree=2)


train_x_poly = poly.fit_transform(train_x)
# Fit transform takes our x values and outputs a list of our data raised from power 0 tot power 2
# (See lab notes for more information)


clf = linear_model.LinearRegression()
train_y = clf.fit(train_x_poly,train_y)

# The coefficients
print('Coefficients: ',clf.coef_)
print('Intercept: ',clf.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
XX = np.arange(0.0,10.0,0.1)
yy = clf.intercept_ + clf.coef_[1]*XX + clf.coef_[2]*np.power(XX,2)
plt.plot(XX,yy,'-r')

plt.xlabel('Engine Size')
plt.ylabel('Emission')


#Evalulation

from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(text_x_poly)

