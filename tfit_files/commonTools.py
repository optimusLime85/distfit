"""
    a set of functions that might normally be available through extra packages
    such as SciPy but that are not easily added to the python that comes with ABAQUS
"""

import math
import csv

def sigmoidLog(x, yMax, xLim, pLim=0.99):
    """ logistic sigmoid function that will reach pLim(%) of yMax at xLim. Plim must be less than 1 (this is not checked) """
    alpha = -math.log(1/pLim-1)/xLim
    return yMax/(1 + math.exp(-alpha*x))


def find_nearest_in_array(array,value):
    """ 
        An efficient search to get the nearest value in a large array 
        Ref: http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    """
    idx = np.searchsorted(array, value, side="left")

    if idx == 0: # first handle end cases
        nearest = array[0]
    elif idx == len(array):
        nearest = array[-1]
    elif math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]): # check if closest point is to the left or right
        nearest = array[idx-1]
    else:
        nearest = array[idx]

    return nearest


def isOdd(x):
    return x % 2 != 0 


def isEven(x):
    return x % 2 == 0


def normalizeAngle(angle):
    """ force angle into range 0 <= angle < 2*pi (0 <= angle < 360 deg) """
    # reduce the angle  
    angle =  angle % (2*math.pi); 

    # force it to be the positive remainder, so that 0 <= angle < 360  
    angle = (angle + 2*math.pi) % (2*math.pi);

    return angle


def magnitude(A):
    """ magnitude of a 3D vector, includes error handling for input = None """
    if A == None:
        mag = None
    else:
        mag = (A[0]**2 + A[1]**2 + A[2]**2)**.5
    return mag


def dist3D(A, B):
    return math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2 + (A[2]-B[2])**2)


def average(A):
    return sum(A)/float(len(A)) # use float to ensure floating point division is used


def linspace(a, b, n=100):
    """ similar behaviour to MatLab's linspace """
    if n < 2:
        linList = b
    diff = (float(b) - a)/(n - 1)
    linList = [diff * i + a  for i in range(n)]
    return linList


def removeDuplicates(seq): 
   """
        removes duplicates from a list or other sequence and
        preserves the original order with a very efficient implementation
        based on f5 here: http://www.peterbe.com/plog/uniqifiers-benchmark
        returns a list
   """
   seen = {}
   result = []
   for entry in seq:
       if entry not in seen: 
           seen[entry] = 1
           result.append(entry)
   return result


def isBetween(x, bound1, bound2):
    """ test is inclusive of the two bounds """
    if (x >= bound1 and x <= bound2) or (x >= bound2 and x <= bound1):
        isBetween = True
    else:
        isBetween = False
    return isBetween


def interpolate(Xint, X, Y):
    """
        Performs a linear interpolation within the range of X and flat extrapolation outside the bounds
        X should change monotonically - this is not checked!
    """

    if Xint <= min(X):
        if X[0] == min(X): # use the upper or lower bound Y value, as appropriate
            Yint = Y[0]
        else:
            Yint = Y[-1]
    elif Xint >= max(X):
        if X[0] == max(X): # use the upper or lower bound Y value, as appropriate
            Yint = Y[0]
        else:
            Yint = Y[-1]
    else:
        i = 0
        while (i < len(X)-1) and not(isBetween(Xint, X[i], X[i+1])):
            i += 1

        Yint = Y[i] + (Xint-X[i])*(Y[i+1]-Y[i])/(X[i+1]-X[i])

    return Yint


def readCSV(fileName, nCol, nHeaderRows = 1):
    """
        reads table of data from a CSV file,
        all data stored in a list of lists
        intended for files with each column representing a variable
        access data by: data[iVariable][iRow] -> data[4][2] will access the data in col 5, row 3 (header rows removed)
    """
    
    if fileName[-4:] != '.csv':
        fileName += '.csv'
    
    dataFile = open(fileName, 'r')  # open file for reading
    dataCSVFile = csv.reader(dataFile)

    allData = []
    for iCol in range(nCol):    # create empty arrays for each column all in a single list
        allData.append([])
    
    for line in dataCSVFile:
        for iCol in range(nCol):
            allData[iCol].append(line[iCol])

    for iRow in range(nHeaderRows): # delete header rows
        for iCol in range(nCol):    
            del allData[iCol][0]

    dataFile.close()

    return allData
    
    
def readCSVwithHeaders(fileName):
    """
        reads table of numerical data from a CSV file,
        all data stored in a dictionary of lists of floats
        the heading in the first row is used as the key for each list
        intended for files with each column representing a variable
    """
    
    if fileName[-4:] != '.csv':
        fileName += '.csv'
    
    dataFile = open(fileName, 'r')  # open file for reading
    dataCSVFile = csv.reader(dataFile)

    allData = {}
    readHeaders = True
    headers = []
    for line in dataCSVFile:
        if readHeaders:
            for entry in line:
                allData[entry] = []     # initiate each list
                headers.append(entry)   # store each header so that we can add the data in the right order below 
            readHeaders = False        
        
        else:
            emptyLine = True           # first determine if the whole line is empty or only contains non-floatable entries
            for iCol in range(len(line)):
                try:
                    float(line[iCol])
                    emptyLine = False  # we could get this far so at least one column has a data point that can convert to a float
                except:
                    pass
                    
            if not(emptyLine):          # copy in the data that we can 
                for iCol in range(len(line)):
                    try:
                        allData[headers[iCol]].append(float(line[iCol]))
                    except:
                        allData[headers[iCol]].append(None)

    dataFile.close()

    return allData



def readSettingsCSV(fileName):
    """
        reads table of user settings from a CSV file
        data stored in a dictionary, similar to hash tables in other languages
    """
    dataFile = open(fileName+'.csv', 'r')  # open file for reading
    dataCSVFile = csv.reader(dataFile)

    settings = {}   ## create empty dictionary
    for line in dataCSVFile:
        if line[0] != '': # skip empty lines
            settings[line[0].strip()] = line[1].strip()  # create entry # .strip() removes leading and trailing whitespace

    dataFile.close()

    return settings


def showSettings(settings):
    """ shows the settings that were read in using readSettingsCSV  - configured for python 3.x """
    print('Settings read in from file:')
    for settingName in list(settings.keys()):
        print(settingName, ' = ', settings[settingName])
    print()
    return


def main():
    X = [0, 1, 11]
    Y = [-1, 0, 100]

    print(interpolate(0.5, X, Y))
    print(interpolate(10, X, Y))
    
    print(3, 'odd', isOdd(3))
    print(3, 'even', isEven(3))
    print(2, 'odd', isOdd(2))
    print(2, 'even', isEven(2))
    print(0, 'odd', isOdd(0))
    print(0, 'even', isEven(0))
    

if __name__ == "__main__":  # used to ensure that main() is only called if this script is run directly
    main()
