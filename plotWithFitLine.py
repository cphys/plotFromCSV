import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import os

def printListOfHeaders(pathToCsv):
    df = pd.read_csv(pathToCsv)
    return print('\nAvalable headers are:\n' + '%s' % '\n'.join(map(str, df.columns.values)) + '\n')



def scatPlot(xAxis,yAxis, pathToCsv, fitline=True, flOrder = 1, xLabel=True, yLabel=True, scale = 7, ratio = 1, fontSize = 20, saveFile = False, dirName = os.getcwd()):

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

    ax.scatter(xVals, yVals,  color='blue')

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
            ax.text(243,53,'$y=$' + fitEqn + '\n$R^2={r:.3f}$'.format(*coefs,r=rSqrd), color='black', fontsize = fontSize, bbox = props)

        else:
            coefs = np.polyfit(xVals, yVals, flOrder)
            lin = np.poly1d(coefs)
            ax.plot(xValsSorted, lin(xValsSorted), '-g')
 

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



scatPlot('CO2EMISSIONS','FUELCONSUMPTION_COMB_MPG','example.csv', flOrder = 1, saveFile = True)

