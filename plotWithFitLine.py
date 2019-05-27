import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

def printListOfHeaders(pathToCsv):
    df = pd.read_csv(pathToCsv)
    return print('\nAvalable headers are:\n' + '%s' % '\n'.join(map(str, df.columns.values)) + '\n')



def scatPlot(xAxis,yAxis, pathToCsv, fitline=True, flOrder = 1, xLabel=True, yLabel=True):
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

    plt.scatter(xVals, yVals,  color='blue')
    print(type(xVals))


    if fitline:
        xValsSorted = np.sort(xVals)
        intOrder = int(flOrder)
        if intOrder == 1:
            coefs = np.polyfit(xVals, yVals, intOrder)
            lin = np.poly1d(coefs)

            ## calculate the r-squrared value ##
            sumListy = np.sum(yVals)
            ybar = np.mean(yVals)
            lxly = np.dot(xVals,yVals)
            lyly = np.dot(yVals,yVals)
            ssR = lyly - coefs[1] * sumListy - coefs[0] * lxly
            ssT = lyly - sumListy * ybar
            rSqrd = 1-ssR/ssT

            plt.plot(xValsSorted, lin(xValsSorted), '-r')

        else:
            coefs = np.polyfit(xVals, yVals, flOrder)
            lin = np.poly1d(coefs)
            plt.plot(xValsSorted, lin(xValsSorted), '-g')  

    if xLabel==True:
        plt.xlabel(str(xAxis))
    elif xLabel==False:
        plt.xlabel(None)
    else:
        plt.xlabel(str(xLabel).title())

    if yLabel==True:
        plt.ylabel(str(yAxis))
    elif yLabel==False:
        plt.ylabel(None)
    else:
        plt.ylabel(str(yLabel).title())

    plt.show()

printListOfHeaders('example.csv')
scatPlot('CO2EMISSIONS','FUELCONSUMPTION_COMB_MPG','example.csv', flOrder = 4)

