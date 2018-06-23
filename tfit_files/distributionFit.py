import os
import sys

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
# import matplotlib.pyplot as plt
# plt.style.use('CFER_plotStyle') # place file CFER_plotStyle.mplstyle in the same directory

from tfit_files import commonTools as common

"""
A generic Distribution class to handle a consistent set of calls for various typically used distributions
Functions are provided for fitting a set of data to a distribution and generating Q-Q type plots, some of which 
mimic the style created by C-FIT

2016.01.06  First rough draft

2016.10.10  neaten up Distribution class to only check user input of distribution type once
            Move distribution fitting code in main to outside function so that it is callable by outside scripts
            Create functions to standardize plots
            Added function fitDistribution to use optimisation to fit the distribution in the probability paper domain

2016.01.11  Modified function fitDistribution to use optimisation to either use the scipy built-in MLE, match the moments of the data or 
            minimise the sqare error in the probability paper domain (i.e. compare data values to distribution at filliben estimate quantile positions)
            
2016.01.20  Add flags to be able to force 0 for location variable

2016.01.23  moved fitDistribution() function into Distribution class and renamed to fit()

"""


class Distribution:
    def __init__(self, data, distribType, minAtZero=False):
        """ minAtzero only applied to log normal, weibull and exponential distributions """
        # self.mean = np.mean(data)
        # self.stdDev = np.std(data, ddof=1) # ddof = 1 for sample std dev

        self.fixedAtZero = minAtZero

        if distribType.lower() == 'normal':
            self.type = 'normal'
        elif distribType.lower() in ['log-normal', 'lognormal', 'log normal']:
            self.type = 'lognormal'
        elif distribType.lower() == 'weibull':
            self.type = 'weibull'
        elif distribType.lower() in ['exponential', 'expon', 'exponent']:
            self.type = 'exponential'
        elif distribType.lower() in ['triangular', 'triang', 'trian']:
            self.type = 'triangular'
        elif distribType.lower() == 'uniform':
            self.type = 'uniform'
        elif distribType.lower() == 'gamma':
            self.type = 'gamma'
        else:
            print('ERROR: distribution type not recognised')

        self.initParams(data)
        self.setDistObj()

        return

    def initParams(self, data):
        """ set the shape, location and scale parameters; uses MLE for everything but the simple distributions (i.e. normal, uniform, triangular) """
        if self.type == 'normal':
            self.shape = 0  # not used; included for optimiser in fitDistribution
            self.loc = np.mean(data)
            self.scale = np.std(data, ddof=1)
            self.nParams = 2

        elif self.type == 'lognormal':
            self.setParamsMLE(data)
            self.nParams = 3
            if self.fixedAtZero: self.nParams -= 1

        elif self.type == 'weibull':
            self.setParamsMLE(data)
            self.nParams = 3
            if self.fixedAtZero: self.nParams -= 1

        elif self.type == 'exponential':
            self.setParamsMLE(data)
            self.nParams = 2
            if self.fixedAtZero: self.nParams -= 1

        elif self.type == 'triangular':
            self.min = np.min(data)
            self.max = np.max(data)
            self.loc = self.min
            self.scale = self.max - self.min
            self.mode = stats.mode(data)
            if self.mode[1][
                0] != 1.0:  # we can only use the mode for the distribution shape if all values are not unique
                self.mid = self.mode[0][0]
            else:  # default to the median instead
                self.mid = np.median(data)

            self.shape = (self.mid - self.min) / self.scale
            self.nParams = 3

        elif self.type == 'uniform':
            self.min = np.min(data)
            self.max = np.max(data)
            self.loc = self.min
            self.scale = self.max - self.min
            self.shape = 0  # not used; included for optimiser in fitDistribution
            self.nParams = 2

        elif self.type == 'gamma':
            self.setParamsMLE(data)
            self.nParams = 3
            if self.fixedAtZero: self.nParams -= 1

        return

    def setParamsMLE(self, data):
        """ set the shape, location and scale parameters using stats library MLE """
        if self.type == 'normal':
            args = stats.norm.fit(data)  # uses MLE

            self.shape = 0  # not used; included for optimiser in fitDistribution
            self.loc = args[0]
            self.scale = args[1]

        elif self.type == 'lognormal':
            if not self.fixedAtZero:
                args = stats.lognorm.fit(data)  # uses MLE
            else:
                args = stats.lognorm.fit(data, floc=0)  # uses MLE

            self.shape = args[0]
            self.loc = args[1]
            self.scale = args[2]

        elif self.type == 'weibull':
            if not self.fixedAtZero:
                args = stats.weibull_min.fit(data)  # uses MLE
            else:
                args = stats.weibull_min.fit(data, floc=0)  # uses MLE

            self.shape = args[0]
            self.loc = args[1]
            self.scale = args[2]

        elif self.type == 'exponential':
            if not self.fixedAtZero:
                args = stats.expon.fit(data)  # uses MLE
            else:
                args = stats.expon.fit(data, floc=0)  # uses MLE

            self.shape = 0  # not used; included for optimiser in fitDistribution
            self.loc = args[0]
            self.scale = args[1]

        elif self.type == 'triangular':
            args = stats.triang.fit(data)  # uses MLE

            self.shape = args[0]
            self.loc = args[1]
            self.scale = args[2]

        elif self.type == 'uniform':
            args = stats.uniform.fit(data)  # uses MLE

            self.shape = 0  # not used; included for optimiser in fitDistribution
            self.loc = args[0]
            self.scale = args[1]

        elif self.type == 'gamma':
            if not self.fixedAtZero:
                args = stats.gamma.fit(data)  # uses MLE
            else:
                args = stats.gamma.fit(data, floc=0)  # uses MLE

            self.shape = args[0]
            self.loc = args[1]
            self.scale = args[2]

        return

    def setDistObj(self):
        """ create and freeze the distribution object from the stats module with the current scale, locationa nd shape parameters """
        if self.type == 'normal':
            self.distObj = stats.norm(loc=self.loc, scale=self.scale)

        elif self.type == 'lognormal':
            self.distObj = stats.lognorm(s=self.shape, loc=self.loc, scale=self.scale)

        elif self.type == 'weibull':
            self.distObj = stats.weibull_min(c=self.shape, loc=self.loc, scale=self.scale)

        elif self.type == 'exponential':
            self.distObj = stats.expon(loc=self.loc, scale=self.scale)

        elif self.type == 'triangular':
            self.distObj = stats.triang(c=self.shape, loc=self.loc, scale=self.scale)

        elif self.type == 'uniform':
            self.distObj = stats.uniform(loc=self.loc, scale=self.scale)

        elif self.type == 'gamma':
            self.distObj = stats.gamma(a=self.shape, loc=self.loc, scale=self.scale)

        #         elif self.type == 'constant':
        #             self.value = float(settings[self.name + '_' + 'value'])

        return

    def pdf(self, x):
        """ exernal access to the distribution probability density function """
        return self.distObj.pdf(x)

    def ppf(self, x):
        """ exernal access to the distribution probability percentile function (the inverse of the cdf) """
        return self.distObj.ppf(x)

    def cdf(self, x):
        """ exernal access to the distribution cumulative probability function """
        return self.distObj.cdf(x)

    def moments(self):
        """ list containing the mean, var and non-central skewness (i.e. the third moment) """
        #    [mean, var, skew] # self.distObj.stats(moments='mvs')
        # [self.distObj.moment(1), self.distObj.moment(2), self.distObj.moment(3)]
        return [self.distObj.mean(), self.distObj.std(), self.distObj.moment(3)]

    def ppf_defaults(self, x):
        """ ppf function but with scale =1 and loc = 0.  Use shape same shape parameters as actual distribution """
        if self.type == 'normal':
            ppf = stats.norm.ppf(x, loc=0, scale=1)

        elif self.type == 'lognormal':
            ppf = stats.lognorm.ppf(x, s=self.shape, loc=0, scale=1)

        elif self.type == 'weibull':
            ppf = stats.weibull_min.ppf(x, c=self.shape, loc=0, scale=1)

        elif self.type == 'exponential':
            ppf = stats.expon.ppf(x, loc=0, scale=1)

        elif self.type == 'triangular':
            ppf = stats.triang.ppf(x, c=self.shape, loc=0, scale=1)

        elif self.type == 'uniform':
            ppf = stats.uniform.ppf(x, loc=0, scale=1)

        elif self.type == 'gamma':
            ppf = stats.gamma.ppf(x, a=self.shape, loc=0, scale=1)

        return ppf

    def fit(self, data, fit='quantiles'):
        """ 
            fit = 'MLE'         - use stats library built in MLE fitting
            fit = 'quantiles'   - use optimisation to find the best fit distribution in the probability 
                                paper domain (i.e. compare data values to distribution at filliben estimate quantile positions)
            fit = 'MOM'         - use optimisation to match the distibution moments to the data moments (this is crude and could be replaced with closed-form equations for some simpler distributions)
                                It would be better to use closed form solutions for some of the simpler distributions. e.g. LogNormal, ref: http://mathforum.org/kb/thread.jspa?forumID=231&threadID=504603&messageID=1541779
            
        """
        if fit == 'MLE':
            self.setParamsMLE(data)
            self.setDistObj()
            isConverged = True  # assume stats.fit will always return a distribution
        else:
            dataMoments = np.array([np.mean(data), np.std(data, ddof=1), moment(data, 3)])

            def objFunc(X):
                [self.shape, self.loc, self.scale] = X
                if self.fixedAtZero:
                    self.loc = 0
                self.setDistObj()
                if fit == 'quantiles':
                    obj = probPlotSqrErr(data, self, self.type, showPlots=False)[0]
                elif fit == 'MOM':
                    distMoments = self.moments()
                    weights = [1, 1,
                               0.1]  # scale the influence of each moment # set last entry to remove skewness from the assessment
                    # scale each moment error relative to the data moment value, but replace the data moment with a constant if it is close to zero
                    obj = np.sum([abs(dataMoments[i] - distMoments[i]) / max(dataMoments[i], 1E-6) * weights[i] for i in
                                  range(
                                      self.nParams)])  # only use the number of moments needed to specify the distribution to match the data # np.sum((distMoments-dataMoments)**2) # np.sum([abs( (dataMoments[i]-distMoments[i])**(1/(i+1)) ) for i in range(3)]) #np.sum((dist.moments()-dataMoments)**2)
                return obj

            X = [self.shape, self.loc, self.scale]

            res = minimize(objFunc, X, method='SLSQP', options={'disp': True, 'maxiter': 600,
                                                                'ftol': 1e-8})  # , bounds=bnds, constraints=cons, # options={'maxiter': 500, 'gtol': 1e-6, 'disp': True}
            # method='SLSQP' 'TNC' 'L-BFGS-B' 'COBYLA' #
            # seems to ignore the constraint if bounds not included with method='SLSQP'
            isConverged = res.success
            if isConverged:
                [self.shape, self.loc, self.scale] = res.x
            else:
                [self.shape, self.loc, self.scale] = X  # revert to previous values

            if self.fixedAtZero:
                self.loc = 0

            self.setDistObj()
        return isConverged


def moment(X, order):
    """ non-central moment of a sample X """
    moment = np.sum([x ** order for x in X]) / len(X)
    return moment


def probPlotSqrErr(data, dist, distributionType, xTitle='Value', fitType='defaults', runID='', outPath='',
                   showPlots=True):
    """ 
        Generate the theoretical and data quantiles to produce probability plots 
        Compares the predicted values to the actual data to determine how well the distribution fits the data in the probabilty paper domain
        
        data is a 1D array
        dist is a Distribution class object
        distributionType is a string used for plot axes titles
        runID is a string for the naming of saved files; not used if showPlots=False
        outPath is a string to specify save location; not used if showPlots=False
    """
    # process
    nData = len(data)

    # generate comparisons on probability paper
    # using filliben's estimate for the median statistics of the theoretical quantiles
    # ref: http://www.real-statistics.com/tests-normality-and-symmetry/graphical-tests-normality-symmetry/ 
    # ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html
    percentiles = np.zeros(nData)
    percentiles[0] = 1 - 0.5 ** (1 / nData)
    percentiles[1:-1] = [(i - 0.3175) / (nData + 0.365) for i in range(2, nData)]
    percentiles[-1] = 0.5 ** (1 / nData)

    dataSorted = np.sort(data)
    distQuantiles = dist.ppf(percentiles)
    sqrErr = np.sum((dataSorted - distQuantiles) ** 2)  # establish how well the distribution predicts the data values

    dataTheorQuantiles = dist.ppf_defaults(percentiles)

    if showPlots:
        dataAlpha = 0.5
        lineColors = colorPalette()

        # histogram
        fig = plt.figure()
        binsHist = np.linspace(np.min(data), np.max(data), 21)
        binsDist = np.linspace(np.min(data) - 0.1 * (np.max(data) - np.min(data)),
                               np.max(data) + 0.1 * (np.max(data) - np.min(data)), 300)
        distPDF = dist.pdf(binsDist)

        plt.hist(data, binsHist, normed=True, color=lineColors[0], edgecolor="none", rwidth=0.85, label='Data')
        plt.plot(binsDist, distPDF, '-', color=lineColors[1], label=distributionType)
        plt.xlabel(xTitle)
        plt.ylabel('Probability Density')
        plt.grid(False)
        plt.savefig(outPath + '\\' + runID + '_' + distributionType + '_' + fitType + '_' + 'histogram' + '.png',
                    dpi=300, format='png')

        # Probability plots
        plotPercentiles = np.array([1E-6, 0.1, 1, 5, 10, 20, 30, 50, 70, 80, 90, 95, 99, 99.9, (100 - 1E-6)]) / 100
        yAxisMin = dist.ppf_defaults(0.01)
        yAxisMax = dist.ppf_defaults(0.99)
        logXaxisMin = 1E-5
        logYaxisMin = dist.ppf_defaults(0.001)

        #         iStart = np.searchsorted(plotPercentiles, percentiles[0])
        #         iEnd = np.searchsorted(plotPercentiles, percentiles[-1])
        #         plotPercentiles = plotPercentiles[max(iStart-1, 0):min(iEnd+1, len(plotPercentiles)-1)] # limit the range of the plot to the closest pair of points to bracket the data percentiles

        distTheorQuantilesPlot = dist.ppf_defaults(plotPercentiles)
        distQuantilesPlot = dist.ppf(plotPercentiles)

        # Q-Q plot; data on x-axis
        fig = plt.figure()
        plt.plot(dataSorted, dataTheorQuantiles, '.', color=lineColors[0], label='Data', alpha=dataAlpha)
        ax = plt.axes()
        xLimCurr = ax.get_xlim()
        yLimCurr = ax.get_ylim()  # get the matplotlib generated axis limits before adding the distribution straight line which will stretch too far

        plt.plot(distQuantilesPlot, distTheorQuantilesPlot, '-', color='k', label=distributionType)
        ax.set_xlim(xLimCurr)  # reset back to limits determined by the range of the data
        ax.set_ylim([min(yLimCurr[0], yAxisMin), max(yLimCurr[1],
                                                     yAxisMax)])  # reset back to limits determined by the range of the data unless it needs to go wider to get to yAxisMin or yAxisMax

        plt.xlabel(distributionType + ' ' + 'Data Quantiles')
        plt.ylabel(distributionType + ' ' + 'Theoretical Quantiles')

        plt.savefig(outPath + '\\' + runID + '_' + distributionType + '_' + fitType + '_' + 'QQ' + '.png', dpi=300,
                    format='png')

        # Q-Q plot; data on x-axis and CDF tick marks on y-axis
        fig = plt.figure()
        plt.plot(dataSorted, dataTheorQuantiles, '.', color=lineColors[0], label='Data', alpha=dataAlpha)
        ax = plt.axes()
        xLimCurr = ax.get_xlim()  # get the matplotlib generated axis limits before adding the distribution straight line which will stretch too far
        yLimCurr = ax.get_ylim()

        plt.plot(distQuantilesPlot, distTheorQuantilesPlot, '-', color='k', label=distributionType)

        plt.xlabel(xTitle)
        plt.ylabel(distributionType + ' ' + 'Distribution Cumulative Probability')

        ax.set_yticks(distTheorQuantilesPlot[1:-1])
        ax.set_yticklabels(plotPercentiles[1:-1])
        ax.set_xlim(xLimCurr)  # reset back to limits determined by the range of the data
        ax.set_ylim([min(min(dataTheorQuantiles), yAxisMin), max(max(dataTheorQuantiles),
                                                                 yAxisMax)])  # reset back to limits determined by the range of the data unless it needs to go wider to get to yAxisMax and yAxisMax

        plt.savefig(outPath + '\\' + runID + '_' + distributionType + '_' + fitType + '_' + 'QQcdf' + '.png', dpi=300,
                    format='png')

        # Q-Q plot; data on x-axis and CDF tick marks on y-axis; log-log scale
        fig = plt.figure()
        Xshift = dist.loc

        plt.plot(dataSorted - Xshift, dataTheorQuantiles, '.', color=lineColors[0], label='Data', alpha=dataAlpha)
        ax = plt.axes()
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')
        xLimCurr = ax.get_xlim()
        yLimCurr = ax.get_ylim()  # get the matplotlib generated axis limits before adding the distribution straight line which will stretch too far

        plt.plot(distQuantilesPlot - Xshift, distTheorQuantilesPlot, '-', color='k', label=distributionType)

        plt.xlabel(xTitle + ' [X - ' + str(format(Xshift, ".4f")) + ']')
        plt.ylabel(distributionType + ' ' + 'Distribution Cumulative Probability')

        ax.tick_params(axis='y', which='minor', left='off')
        ax.tick_params(axis='y', which='minor', right='off')
        ax.set_yticks(distTheorQuantilesPlot[1:-1])
        ax.set_yticklabels(plotPercentiles[1:-1])
        ax.set_xlim(
            [max(xLimCurr[0], logXaxisMin), xLimCurr[1]])  # reset back to limits determined by the range of the data
        ax.set_ylim([max(min(min(dataTheorQuantiles), yAxisMin), logYaxisMin), max(max(dataTheorQuantiles),
                                                                                   yAxisMax)])  # reset back to limits determined by the range of the data unless it needs to go wider to get to yAxisMax - do not stretch below yAxisMin because skew distributions can go very low for x values close to Xshift

        plt.grid(False, which='minor', axis='both')  # , linestyle='-', color='lightgrey')

        plt.savefig(outPath + '\\' + runID + '_' + distributionType + '_' + fitType + '_' + 'QQcdf_loglog' + '.png',
                    dpi=300, format='png')

    return [sqrErr, dist.distObj.mean(), dist.distObj.std(), dist.loc]


def colorPalette():
    colors = [[0, 0.329411764705882, 0.619607843137255],
              [0.552941176470588, 0.776470588235294, 0.247058823529412],
              [0.701960784313725, 0.12156862745098, 0.0549019607843137],
              [0.552941176470588, 0.776470588235294, 0.847058823529412],
              [0.498039215686275, 0.329411764705882, 0.619607843137255],
              [0.854901960784314, 0.505882352941176, 0.215686274509804]]
    return colors


def checkTrue(inputString):
    """ check if user input string = yes """
    if inputString.lower() in ['y', 'yes', 'true']:
        check = True
    else:
        check = False
    return check


def main(currPath, inputFileName, showPlots):
    inPath = currPath + '\\Inputs'
    dataHeader = 'Depth'

    # read in the inputs
    settings = common.readSettingsCSV(inPath + '\\' + inputFileName)
    common.showSettings(settings)
    runID = settings['runID']
    dataFileName = settings['dataFileName']
    distributionType = settings['distribution']
    minAtZero = checkTrue(settings['fixMinAtZero'])
    xTitle = settings['xTitle']

    outPath = currPath + '\\Reports'
    if os.path.isdir(outPath) == False:  # create directory if it doesn't exist
        os.mkdir(outPath)

    # read in the data
    data = common.readCSVwithHeaders(inPath + '\\' + dataFileName)
    data = np.array(data[dataHeader])

    # create the distribution from the data with default initialisation (MLE for most distributions)
    dist = Distribution(data, distributionType, minAtZero)

    # check fit on probability paper
    [sqrErr, mean, stdDev, lowerBound] = probPlotSqrErr(data, dist, distributionType, xTitle, 'MLE', runID, outPath,
                                                        showPlots)  # output: [sqrErr, mean, stdDev, lowerBound]
    print(sqrErr, mean, stdDev ** 2, lowerBound)

    # fit distribution and check fit again on probability paper
    fitType = 'MOM'
    isConverged = dist.fit(data, fit=fitType)  # fit='MLE' # fit='MOM' # fit='quantiles'
    if isConverged:
        [sqrErr, mean, stdDev, lowerBound] = probPlotSqrErr(data, dist, distributionType, xTitle, fitType, runID,
                                                            outPath,
                                                            showPlots)  # output: [sqrErr, mean, stdDev, lowerBound]
        print(sqrErr, mean, stdDev ** 2, lowerBound)
        print('dist.moments():          ', dist.moments())
        print('dist central moments:    ', dist.distObj.stats(moments='mvs'))
        print('data central moments:    ', stats.moment(data, moment=1), stats.moment(data, moment=2),
              stats.moment(data, moment=3))
        print('data non-central moments:', moment(data, 1), moment(data, 2), moment(data, 3))
        print('data mvs:                ', np.mean(data), np.var(data, ddof=1), stats.skew(data))

        # try creating a sample from the fit distribution 
        sample = dist.ppf(np.random.random(1E6))
        print('sample mvs:              ', np.mean(sample), np.var(sample, ddof=1), stats.skew(sample))

        print('data mean stdDev min:    ', np.mean(data), np.std(data, ddof=1), np.min(data))
        print('dist mean stdDev min:    ', mean, stdDev, lowerBound)
    else:
        print('distribution fit did not converge')
    return


if __name__ == "__main__":  # used to ensure that main() is only called if this script is run directly
    # program_name = sys.argv[0]
    arguments = sys.argv[1:]
    if len(arguments) == 0:  # run as script in Python
        currPath = os.getcwd()
        inputFileName = 'settingsDistributionFit'
        showPlots = True  # view plots when running as script in Python
    else:
        currPath = arguments[0]
        inputFileName = arguments[1]
        showPlots = False  # do not create plots for user release version called from the command line

    main(currPath, inputFileName, showPlots)
