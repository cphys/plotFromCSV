import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import os
from sklearn import linear_model
from sklearn.metrics import r2_score
from matplotlib.offsetbox import AnchoredText


def printListOfHeaders(pathToCsv):
    df = pd.read_csv(pathToCsv)
    return print('\nAvalable headers are:\n' + '%s' % '\n'.join(map(str, df.columns.values)) + '\n')


def scatPlot(xAxis,yAxis, pathToCsv, fitline=True, flOrder = 1, xLabel=True, yLabel=True, scale = 5, ratio = .75, fontSize = 10, saveFile = False, dirName = os.getcwd(), framePos = 'tl'):

    fig = plt.figure(figsize=(scale, ratio * scale))
    ax = fig.add_subplot(111)

    df = pd.read_csv(pathToCsv)
    headerLabels = df.columns.values    
    try:
        xVals = df[xAxis]
    except KeyError:
        print('\nError: No column named ' + xAxis + ' is avalable' '\nAvalable columns are:\n' + '%s' % '\n'.join(map(str, headerLabels)) + '\n')
    try:
        yVals = df[yAxis]
    except KeyError:
        print('\nError: No column named ' + yAxis + ' is avalable' '\nAvalable columns are:\n' + '%s' % '\n'.join(map(str, headerLabels)) + '\n')

    ax.scatter(xVals, yVals, alpha=0.75,  color='blue',edgecolors = 'none', s=30)

    ## position of data frame window ##
    #
    #  need to define other positions
    if framePos == 'tl':
        xpos, ypos = [.125,.875]


    if fitline:
        xValsSorted = np.sort(xVals)
        intOrder = int(flOrder)
        if intOrder == 1:
            coefs = np.polyfit(xVals, yVals, intOrder)
            lin = np.poly1d(coefs)

            ## calculate the r-squrared value ##
            sumListy = np.sum(yVals)
            ybar = np.mean(yVals)
            lxly = np.dot(xVals, yVals)
            lyly = np.dot(yVals, yVals)
            ssR = lyly - coefs[1] * sumListy - coefs[0] * lxly
            ssT = lyly - sumListy * ybar
            rSqrd = 1-ssR/ssT

            ax.plot(xValsSorted, lin(xValsSorted), '-r')

            fitEqn = str(lin)[2:]
            props = dict(facecolor = 'white', alpha = 0.6, pad = 8) 
            ax.text(xpos * np.max(xVals),ypos * np.max(yVals),'$y=$' + fitEqn + '\n$R^2={r:.3f}$'.format(*coefs,r=rSqrd), color='black', fontsize = fontSize, bbox = props)

        else:
            coefs = np.polyfit(xVals, yVals, flOrder)
            lin = np.poly1d(coefs)
            ax.plot(xValsSorted, lin(xValsSorted), '-r')
 

    if xLabel==True:
        ax.set_xlabel(str(xAxis).replace("_"," ").title(), fontsize = fontSize)
    elif xLabel==False:
        ax.set_xlabel(None)
    else:
        ax.set_xlabel(str(xLabel).title(),fontsize = fontSize)

    if yLabel==True:
        ax.set_ylabel(str(yAxis).replace("_"," ").title(), fontsize=fontSize)
    elif yLabel==False:
        ax.set_ylabel(None)
    else:
        ax.set_ylabel(str(yLabel).title(), fontsize=fontSize)

    ax.tick_params(axis='both', labelsize=fontSize)

    if saveFile:
        plt.savefig(os.path.join(dirName, '{}_vs_{}_fitOrder{}.png'.format(yAxis,xAxis,flOrder)), bbox_inches='tight')

    plt.show()



def plotMLregression(xAxis, yAxis, pathToCsv, colors = ("blue", "red"), location=2, scale = 5, ratio = .75, fitline = True, flOrder = 1, xLabel=True, yLabel=True, fontSize = 10, saveFile = False, dirName = os.getcwd()):
    fig = plt.figure(figsize=(scale, ratio * scale))
    ax = fig.add_subplot(111)
    df = pd.read_csv(pathToCsv)
    try:
        xVals = df[xAxis]
    except KeyError:
        print('\nError: No column named ' + xAxis + ' is avalable' '\nAvalable columns are:\n' + '%s' % '\n'.join(map(str, headerLabels)) + '\n')
    try:
        yVals = df[yAxis]
    except KeyError:
        print('\nError: No column named ' + yAxis + ' is avalable' '\nAvalable columns are:\n' + '%s' % '\n'.join(map(str, headerLabels)) + '\n')


    #  for the purposes of learning machine learning tools 
    #  we divide the data up into two sets. Here we decide
    #  that we want to use 80% of the data for training and
    #  20% for tessting
    pL = [xAxis,yAxis]
    cdf = df[[xAxis,yAxis]]

    msk = np.random.rand(len(df)) < 0.8
    train = cdf[msk]
    test = cdf[~msk]

    pl1 = (train[xAxis], train[yAxis])
    pl2 =  (test[xAxis], test[yAxis])

    data = (pl1,pl2)    
    groups = ("Test Data", "Training Data")
    ax = fig.add_subplot(111)
    for data, color, group in zip(data, colors, groups):
        x, y = data
        ax.scatter(x, y, alpha=0.75, c=color, edgecolors='none', s=30, label=group)
    plt.legend(loc=location)

    if fitline:
        xValsSorted = np.sort(xVals)
        intOrder = int(flOrder)
        if intOrder == 1:
            regr = linear_model.LinearRegression()
            train_x = np.asanyarray(train[[xAxis]])
            train_y = np.asanyarray(train[[yAxis]])
            regr.fit (train_x, train_y)

            test_x = np.asanyarray(test[[xAxis]])
            test_y = np.asanyarray(test[[yAxis]])
            test_y_ = regr.predict(test_x)

            # Print the coefficients from the expansion
            print ('\nCoefficients: ', regr.coef_)
            print ('Intercept: ',regr.intercept_)
            # print various error metrics to mesasure the accuracy of our linear model
            print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
            print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
            print("R2-score: %.2f" % r2_score(test_y_ , test_y) )

            ax.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-k')

        

    if xLabel==True:
        ax.set_xlabel(str(xAxis).replace("_"," ").title(), fontsize = fontSize)
    elif xLabel==False:
        ax.set_xlabel(None)
    else:
        ax.set_xlabel(str(xLabel).title(),fontsize = fontSize)

    if yLabel==True:
        ax.set_ylabel(str(yAxis).replace("_"," ").title(), fontsize=fontSize)
    elif yLabel==False:
        ax.set_ylabel(None)
    else:
        ax.set_ylabel(str(yLabel).title(), fontsize=fontSize)

    ax.tick_params(axis='both', labelsize=fontSize)

    if saveFile:
        plt.savefig(os.path.join(dirName, 'mL_{}_vs_{}_fitOrder{}.png'.format(yAxis,xAxis,flOrder)), bbox_inches='tight')
        
    plt.show()

def histPlots(paramList,pathToCsv):
    df = pd.read_csv(pathToCsv)
    cdf = df[paramList]
    ## Visualize the data by creating a histogram of each data set ##
    cdf.hist()
    plt.show()
