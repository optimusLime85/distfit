import os
import sys
from copy import deepcopy
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import commonTools as common
import distributionFit as dfit


def checkTrue(inputString):
    """ check if user input string = yes """
    if inputString.lower() in ['y', 'yes', 'true']:
        check = True
    else:
        check = False
    return check

def colorPalette():
    colors = [[0, 0.329411764705882, 0.619607843137255],
              [0.552941176470588, 0.776470588235294, 0.247058823529412],
              [0.701960784313725, 0.12156862745098, 0.0549019607843137],
              [0.552941176470588, 0.776470588235294, 0.847058823529412],
              [0.498039215686275, 0.329411764705882, 0.619607843137255],
              [0.854901960784314, 0.505882352941176, 0.215686274509804]]
    return colors


def sampleWithReplacement(obs, prob, nSim=1E4):
    """
        Draw a set from obs with replacement
        prob is the probabilty that each entry could occur
    """
    ind = [i for i in range(len(obs))] # all indices of obs
    indRand = np.random.choice(ind, size=nSim, replace=True, p=prob) # randomly pick from ind with probabilities per entry of prob
    sample = obs[indRand] # retrieve the actual values from obs
    
    return sample

def detFuncConst(d_th, POD_th):
    """ detection function constant """
    q = -np.log(1-POD_th)/d_th
    return q
    
def POD(reportedDepth, detThreshold, PODthreshold):
    """ The probability of detection given the reported depth """
    q = detFuncConst(detThreshold, PODthreshold)
    pod = 1 - np.exp(-q*reportedDepth)
    return pod


def gen_nonReportedSample(reportedSize, relFreq):
    """ 
        return a randomly generated sample of a representative population of non-detected defects
    """
    nonReportedSample = sampleWithReplacement(reportedSize, relFreq, nSim=1E4)
    return nonReportedSample


def nonReportedParams(reportedDepth, detThreshold, PODthreshold, POFC, POI, useConstPOD, constPOD):
    """ 
        get the likelihood of non-reported defects for each reported defect; the relative frequency and the total expected number of non reported defects
    """
    if useConstPOD:
        reportedPOD = np.array([constPOD]*len(reportedDepth))
    else:
        reportedPOD = POD(reportedDepth, detThreshold, PODthreshold) # POD is a function of depth only
    likelihood = (1-POFC)*(1/(POI*reportedPOD)-1) #(1 - POD) / POD
    nNonReported = np.sum(likelihood)
    relFreq = likelihood/nNonReported # normalise the likelihoods to get at relative probabilities
    
    return [nNonReported, likelihood, relFreq]


def plotSummary(reportedSize, nonReportedSample, dist, distributionType, fitType, xTitle, regionID, outPath):
    fig = plt.figure()
    lineColors = colorPalette()
    binsHist = np.linspace(np.min(reportedSize), np.max(reportedSize), 21)
    binsDist = np.linspace(np.min(reportedSize) - 0*(np.max(reportedSize)-np.min(reportedSize)), np.max(reportedSize) + 0.1*(np.max(reportedSize)-np.min(reportedSize)), 300) 
    distPDF = dist.pdf(binsDist)

    plt.hist([reportedSize, nonReportedSample], binsHist, normed=True, color=[lineColors[0], lineColors[1]], edgecolor = "none", rwidth = 0.85, label=['Reported', 'Unreported'], alpha = 0.5)
    plt.plot(binsDist, distPDF, '-', color=lineColors[1], label='Unreported' + ' ' + distributionType)
    plt.xlabel(xTitle)
    plt.ylabel('Probability Density')
    plt.grid(False) #, which='minor', axis='both')
    plt.legend(loc='upper right')
    plt.savefig(outPath + '\\' + regionID + '_' + distributionType + '_' + fitType + '_' + 'summary' + '.png', dpi = 300, format = 'png')
    return


def processRegion(reportedSize, relativeFrequency, distributionType, minAtZero, defectDimension, plotUnits, regionID, outPath, showPlots, useData):
    """ get representative sample of non-reported distribution and use it to fit a distribution """
    xTitle = defectDimension + ' ' + '(' + plotUnits + ')'
    fitTypes = ['MLE', 'quantiles', 'MOM']
    summaryList = '' # a string prepared for output to a csv file
    
    if useData:
        nonReportedSample = reportedSize
    else:
        nonReportedSample = gen_nonReportedSample(reportedSize, relativeFrequency)
    dist = dfit.Distribution(nonReportedSample, distributionType, minAtZero)
    for s in [np.mean(nonReportedSample), np.std(nonReportedSample, ddof=1), np.min(nonReportedSample)]: summaryList += str(s)+','
    
    for fitType in fitTypes:
        if useData:
            isConverged = dist.fit(reportedSize, fit=fitType)
        else:
            isConverged = dist.fit(nonReportedSample, fit=fitType)
        if isConverged:
            [sqrErr, mean, stdDev, lowerBound] = dfit.probPlotSqrErr(nonReportedSample, dist, distributionType, xTitle, fitType, regionID, outPath, showPlots) # output: [sqrErr, mean, stdDev, lowerBound]
            print(fitType, 'sqrErr:', sqrErr)
            print('mean, stdDev, lower Bound')
            print('dist:', mean, stdDev, lowerBound, sep='\t')
            print('data:', np.mean(nonReportedSample), np.std(nonReportedSample, ddof=1), np.min(nonReportedSample), sep='\t')
            if showPlots:
                plotSummary(reportedSize, nonReportedSample, dist, distributionType, fitType, xTitle, regionID, outPath)
        else: 
            [sqrErr, mean, stdDev, lowerBound] = np.nan*np.ones(4)
        for s in [sqrErr, mean, stdDev, lowerBound]: summaryList += str(s)+','
    return summaryList

def main(currPath, inputFileName, showPlots):
    inPath = currPath + '\\Inputs'
    
    # read in the inputs
    settings = common.readSettingsCSV(inPath + '\\' + inputFileName)
    common.showSettings(settings)
    runID = settings['runID']
    defectFileName = settings['defectFileName']
    regionsFileName = settings['regionsFileName']
    units = settings['units']                           # metric | imperial
    processLength = checkTrue(settings['processLength'])
    processWidth = checkTrue(settings['processWidth'])
    distributionType = settings['distribution']
    minAtZero = checkTrue(settings['fixMinAtZero'])
    POFC = float(settings['POFC'])/100                  # given as % in input file
    POI = float(settings['POI'])/100                    # given as % in input file
    PODthreshold = float(settings['PODthreshold'])/100  # given as % in input file
    useConstPOD = checkTrue(settings['useConstPOD'])
    constPOD = float(settings['constPOD'])
    useData = checkTrue(settings['useData'])
    
    detThresholdByDefect = checkTrue(settings['detThresholdByDefect'])
    
    if units.lower() in ['imperial', 'imp', 'oil field']:
        defectUnits = 'in'
        postUnits = 'ft'
        lengthUnits = 'mi'
        lengthScale = 5280.
    elif units.lower() in ['metric', 'si']:
        defectUnits = 'mm'
        postUnits = 'km'
        lengthUnits = 'km'
        lengthScale = 1.
    else:
        print('ERROR: units identifier not recognised')
        

    outPath = currPath + '\\Reports'
    if os.path.isdir(outPath)==False:   # create directory if it doesn't exist
        os.mkdir(outPath)

    # read in the data
    regions = common.readCSVwithHeaders(inPath + '\\' + regionsFileName)
    regionStart = np.array(regions['Start Post'])   # ft | km
    regionEnd = np.array(regions['End Post'])       # ft | km
    regionNums = regions['Region Number'] # expecting integers
    
    defectData = common.readCSVwithHeaders(inPath + '\\' + defectFileName)
    defectPost = np.array(defectData['Post'])   # station along the line (km, ft)
    reportedDepthAll = np.array(defectData['Depth'])
    if processLength: reportedLengthAll = np.array(defectData['Length'])
    if processWidth: reportedWidthAll = np.array(defectData['Width'])
    
    if detThresholdByDefect:
        detThresholdAll = np.array(defectData['Detection Threshold'])                      # same units as depth
    else:
        detThresholdAll = float(settings['detThreshold'])*np.ones(len(reportedDepthAll))   # same units as depth
    
    # prepare summary of results
    depthWriteList = []
    depthWriteList.append(','*5 + 'Simulated Sample' + ','*3 + 'MLE' + ','*4 + 'Quantiles' + ','*4 + 'MOM' + ','* 4 + '\n')
    paramHeaders = 'Mean' + ' (' + defectUnits + ')' + ',' + 'Std Dev' + ' (' + defectUnits + ')' + ',' + 'Lower Bound' + ' (' + defectUnits + ')'
    depthWriteList.append('Region Number' + ',' + 'Start' + ' (' + postUnits + ')' + ',' + 'End' + ' (' + postUnits + ')' + ',' + \
                            'Length'  + ' (' + lengthUnits + ')' + ',' + 'Expected Density'  + ' (/' + lengthUnits + ')' + ',' + \
                            paramHeaders + ',' + ('Sqr Error' + ' (' + defectUnits + ')' + ',' + paramHeaders + ',')*3 + '\n')
                            
    if processLength: lengthWriteList = deepcopy(depthWriteList)
    if processWidth:  widthWriteList  = deepcopy(depthWriteList)
    
    for iRegion in range(len(regionStart)):
        regionID = runID + '_' + 'Region' + str(format(regionNums[iRegion], '.0f'))
        
        indInRegion = (defectPost >= regionStart[iRegion]) * (defectPost < regionEnd[iRegion])
        if iRegion == len(regionEnd)-1: # also include defect if the station == the end post for the last region 
            indInRegion += (defectPost == regionEnd[iRegion]) 
        reportedDepth = reportedDepthAll[indInRegion]
        detThreshold  = detThresholdAll[indInRegion]
        nReported = len(reportedDepth)
        nReported
        
        # get the likelihood of non-reported defects for each reported defect; the relative frequency and the total expected number of non reported defects
        [nNonReported, likelihood, relativeFrequency] = nonReportedParams(reportedDepth, detThreshold, PODthreshold, POFC, POI, useConstPOD, constPOD)
        regionLength = (regionEnd[iRegion] - regionStart[iRegion])/lengthScale # convert to miles or kms
        defectDensity = nNonReported/regionLength
        for s in [regionNums[iRegion], regionStart[iRegion], regionEnd[iRegion], regionLength, defectDensity]: depthWriteList.append(str(s)+',')
        if processLength: 
            for s in [regionNums[iRegion], regionStart[iRegion], regionEnd[iRegion], regionLength, defectDensity]: lengthWriteList.append(str(s)+',')
        if processWidth: 
            for s in [regionNums[iRegion], regionStart[iRegion], regionEnd[iRegion], regionLength, defectDensity]: widthWriteList.append(str(s)+',')
        
        if nReported > 1: # need at least 2 defects in this region
            
            defectDimension = 'Depth'
            summaryList = processRegion(reportedDepth, relativeFrequency, distributionType, minAtZero, defectDimension, defectUnits, regionID + '_' + defectDimension, outPath, showPlots, useData)
            depthWriteList.append( summaryList )
            depthWriteList.append('\n')
            plt.close('all') # clean up the display - otherwise this could generate a ludicrous number of plots
            
            if processLength:
                reportedLength = reportedLengthAll[indInRegion]
                defectDimension = 'Length'
                # use this to make assumption that lengths are independant of depth: relativeFrequency = np.ones(len(reportedLength))/float(len(reportedLength))
                summaryList = processRegion(reportedLength, relativeFrequency, distributionType, minAtZero, defectDimension, defectUnits, regionID + '_' + defectDimension, outPath, showPlots, useData)
                lengthWriteList.append( summaryList )
                lengthWriteList.append('\n')
                plt.close('all') # clean up the display - otherwise this could generate a ludicrous number of plots
                
                
            if processWidth:
                reportedWidth = reportedWidthAll[indInRegion]
                defectDimension = 'Width'
                # use this to make assumption that widths are independant of depth: relativeFrequency = np.ones(len(reportedWidth))/float(len(reportedWidth))
                summaryList = processRegion(reportedWidth, relativeFrequency, distributionType, minAtZero, defectDimension, defectUnits, regionID + '_' + defectDimension, outPath, showPlots, useData)
                widthWriteList.append( summaryList )
                widthWriteList.append('\n')
                plt.close('all') # clean up the display - otherwise this could generate a ludicrous number of plots
        else:
            depthWriteList.append('\n')
            if processLength: lengthWriteList.append('\n')
            if processWidth: widthWriteList.append('\n')
    
    # write results to file
    outFile = open(outPath + '\\' + runID + '_' + distributionType + '_' 'SummaryDepth' + '.csv', 'w')
    batchWrite = ''.join(depthWriteList)  
    outFile.write(batchWrite)
    outFile.close()
    
    if processLength:
        outFile = open(outPath + '\\' + runID + '_' + distributionType + '_' 'SummaryLength' + '.csv', 'w')
        batchWrite = ''.join(lengthWriteList)  
        outFile.write(batchWrite)
        outFile.close()
    
    if processWidth:    
        outFile = open(outPath + '\\' + runID + '_' + distributionType + '_' 'SummaryWidth' + '.csv', 'w')
        batchWrite = ''.join(widthWriteList)  
        outFile.write(batchWrite)
        outFile.close()
        
    return


if __name__ == "__main__":  # used to ensure that main() is only called if this script is run directly
    # program_name = sys.argv[0]
    arguments = sys.argv[1:]
    if len(arguments) == 0: # run as script in Python
       currPath = os.getcwd()
       inputFileName = 'settingsNonReported' # 'GyroViewSettings' #
       showPlots = True # view plots when running as script in Python
    else:
       currPath = arguments[0]
       inputFileName = arguments[1]
       showPlots = False # do not create plots for user release version called from the command line

    main(currPath, inputFileName, showPlots)