"""
    a library of useful conversions
    usage:
        import conversions as conv
            for one value, x:  x = conv.FtoC(x)
            for a list, X:     X = [conv.FtoC(x) for x in X] # or you can use a traditional for loop
"""

import math

def daysToSec(x):
    return x*86400.0

def yearsToSec(x):
    return x*365.25*86400.0  # ignoring the 0.25

def secToDays(x):
    return x/86400.0

def FtoC(x):
    return (x-32.0)/1.8

def CtoF(x):
    return x*1.8+32.0

def FtToM(x):
    return x*0.3048

def MtoFt(x):
    return x/0.3048
    
def radToDeg(x):
    return x*180.0/math.pi

def degToRad(x):
    return x*math.pi/180.0