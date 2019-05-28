import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import os
from sklearn import linear_model
from sklearn.metrics import r2_score
from plotWithFitLine import printListOfHeaders
from plotWithFitLine import scatPlot
from plotWithFitLine import plotMLregression
from plotWithFitLine import histPlots
from plotWithFitLine import mlLinRegressParams

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

## Print a list of all of the possible headers in the csv file ##
printListOfHeaders("example.csv")

## Create a parameter list to be looked at with histogram plots ##
pL = ['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']
histPlots(pL,"example.csv")

## Plot of Linear regression using sklearn machine learning ##
plotMLregression('ENGINESIZE','CO2EMISSIONS',"example.csv",saveFile=True)

## Plot of Linear regression using numpy values ##
scatPlot('ENGINESIZE','CO2EMISSIONS',"example.csv",saveFile=True)

## Calculate the  ##
xVals = ['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']
yVals = ['CO2EMISSIONS']
regParams = mlLinRegressParams(xVals, yVals, "example.csv", trainDataRatio = 0.8)
print ('\nmultiple linear regression')
print ('Coefficients: ', regParams[0])
print("Residual sum of squares: %.2f" % regParams[1])
print('Variance score: %.2f' % regParams[2])
