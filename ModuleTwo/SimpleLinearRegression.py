#!/usr/bin/env python
# coding: utf-8

# In[32]:


'''
Aim of the Lab:

1) Learn how to use skikit-learn to implement simple linear regression
2) Download a data set that is related to fuel consumption and Carbon dioxide emission of cars
3) Split this data into training and test sets
4) Create a model using that training set
5) Evaluate the model using the test set
6) Use the model to predict an unknown value
'''

# Import the packages we are going to need
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


# Download the data from the IBM object storage
#!wget --no-check-certificate https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv


# In[41]:


# Take a look at the dataset by creating a dataframe using pandas
df = pd.read_csv("FuelConsumptionCo2.csv")
# Display the first 6 rows of the data set
df.head(6)


# In[42]:


# summarise the data
df.describe()

# describe returns various statistics on our data which may be useful.


# In[44]:


# Explore some more features
# Returns a dataframe with just the requested columns (features)
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(6)


# In[48]:


# plot each of these features using Visulization from panadas
viz = cdf[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()
plt.show()


# In[53]:


# Plot each of these features Vs the Emission to see how linear their relation is:
plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("C02 EMISSION")
plt.show()

# Hence, there is a linear trend between EMISSION and FUELCONSUMPTION_COMB
# In[64]:


''' PRACTICE : Plot CYLINDERS vs the EMISSIONS, to see how linear their relation:'''
plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS,color='red')
plt.xlabel("CYLINDERS")
plt.ylabel("C02 EMISSION")
plt.show()


# In[70]:


''' Create train and test dataset'''

# First step is to split our dataset into train and test sets, 80% of the entire data for training and 20% for testing.
# We create a mask to select random rows using np.random.rand()
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk] # Gets rid of the data points chosen by msk

# We are going to use ENGINESIZE and CO2EMISSIONS as our training data


# In[84]:


''' Modelling '''
from sklearn import linear_model
regr = linear_model.LinearRegression()
# Converts data into an array
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

regr.fit(train_x,train_y)
#The coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)


# In[86]:


''' Plot Outputs'''
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x,regr.coef_*train_x+regr.intercept_,'-r')
plt.ylabel('EMISSION')
plt.xlabel('Engine Size')


# In[89]:


''' Evaluation '''
from sklearn.metrics import r2_score
#Converting our test data to an array format
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

#Test our model by predicting some values through passing in our testdata
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y)**2))
print("R2-score: %.2f" % r2_score(test_y_, test_y))

# The higher the values of R^2 the better the model fits your data. Best possible score is 1.0.


