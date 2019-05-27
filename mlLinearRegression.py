import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from plotWithFitLine import scatPlot
#%matplotlib inline

## Reading in the data and creating a dataframe using pandas ##
# df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv")
df = pd.read_csv("example.csv")

## Save the data set as a csv ##
# df.to_csv('example.csv', index=False)

## Print the first 9 rows in the data set ##
print(df.head(9))

## Print a summary of the data set ##
print(df.describe())

## Creates a list containing all of the column headers in the dataset ##
headerLabels = df.columns.values



def histPlots(paramList):
    cdf = df[paramList]
    ## Visualize the data by creating a histogram of each data set ##
    cdf.hist()
    plt.show()

## Select for only the columns that we will be using ##
pL = ['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']

# histPlots(pL)

# scatPlot('FUELCONSUMPTION_COMB','CO2EMISSIONS','example.csv', flOrder = 1)
# scatPlot('ENGINESIZE','CO2EMISSIONS','example.csv', flOrder = 1)
# scatPlot('CYLINDERS','CO2EMISSIONS','example.csv', flOrder = 1)


########     beginning with machie learning     ########
#  for the purposes of learning machine learning tools 
#  we divide the data up into two sets. Here we decide
#  that we want to use 80% of the data for training and
#  20% for tessting
########################################################
cdf = df[pL]
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


## Plot the training data subset along with the test data subset ##
pl1 = (train.ENGINESIZE, train.CO2EMISSIONS)
pl2 =  (test.ENGINESIZE, test.CO2EMISSIONS)
data = (pl1,pl2)
colors = ("blue", "red")
groups = ("Test Data", "Training Data")
fig = plt.figure()
ax = fig.add_subplot(111)
for data, color, group in zip(data, colors, groups):
    x, y = data
    ax.scatter(x, y, alpha=0.75, c=color, edgecolors='none', s=30, label=group)
plt.legend(loc=2)
plt.show()


# Have sklearn calculate a linear regression on the training data ##
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
# Print the coefficients from the expansion
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

# Here we show the results of the linear regression
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# print various error metrics to mesasure the accuracy of our linear model
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )

# here we show the trainining data in blue, the test data in red
# with the green line representing a linear fit of all data
# and the black line representing a linear fit of just the training data
coefs = np.polyfit(cdf.ENGINESIZE, cdf.CO2EMISSIONS,1)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue', alpha=0.75)
plt.scatter(test.ENGINESIZE, test.CO2EMISSIONS,  color='red', alpha=0.75)
plt.plot(train_x,coefs[0]*train_x+coefs[1],'-g')
plt.plot(train_x, regr.coef_[0,0]*train_x + regr.intercept_[0], '--k')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


