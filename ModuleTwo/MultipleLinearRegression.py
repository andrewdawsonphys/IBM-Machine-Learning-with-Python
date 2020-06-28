import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline


# Import the data set from https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv

df = pd.read_csv('FuelConsumptionCo2.csv')
df.head(5)

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head()

# PLOT EMISSION VALUES AGAINST THE ENGINE SIZE
plt.scatter(cdf.ENGINESIZE,cdf.CO2EMISSIONS, color='blue')
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')
plt.show()

# TRAIN A MULTIPLE LINEAR REGRESSION MODEL USING THE DATASET ABOVE
msk = np.random.rand(len(df)) < 0.8

#Split the data into 80% training data and 20% test data
train = cdf[msk]
test = cdf[~msk]

# In reality there are multiple variables which predict the C02 emissions of a car. When one or more independent variable is precent
# then the process is called MULTIPLE LINEAR REGRESSION.

from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])

# Use the data to produce a regression line. sklearn uses Ordinary Least Squares method to solve this problem
regr.fit(x,y)

#The coefficients (parameters: theta0 theta1 and theta2)
print('Coefficients: {}'.format(regr.coef_))



# PREDICTION

y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])

print("Residual sum of squares: %.2f" % np.mean((y_hat-y)**2))

# Explained variance score: 1 is perfect prediction
print('Varience score: %.2f' % regr.score(x,y))

# Now if we plot FUELCONSUMPTION_CITY ; FUELCONSUMPTION_HWY instead of FUELCONSUMPTION_COMB. Does this result in better accuracy?
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])

regr.fit(x,y)

print('Coefficients: {}'.format(regr.coef_))


y_hat = regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])

print("Residual sum of squares: %.2f" % np.mean((y_hat-y)**2))


print('Varience score: %.2f' % regr.score(x,y))
