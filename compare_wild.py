import numpy as np
import pandas as pd
import glob
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
from xgboost import XGBClassifier, XGBRegressor
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from datetime import datetime, timedelta
from sklearn import datasets, linear_model, manifold, metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import plotly as py
import plotly.graph_objects as go
import inspect
import shutil
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import initializers
from time import time
import pickle
import joblib



def resample(dataDf, timeColHeader, samplingRate, gapTolerance=np.inf, fixedTimeColumn=None):
    """
    Parameters
    ----------
    dataDf : data dataframe, contains unixtime column and data column(s)

    timeColHeader : string, time column header

    samplingRate : int
        Number of samples per second

    gapTolerance: int(ms)
        if the distance between target point and either of the neighbors is further than gapTolerance in millisecond,
        then interpolation is nan
        if gapTolerance=0, the gapTolerance rule will not exist

    fixedTimeColumn:

    Examples
    --------
    >>> timeColHeader = 'unixtime'
    >>> df = pd.DataFrame(np.arange(20).reshape(5,4),
                      columns=['unixtime', 'A', 'B', 'C'])

    >>> unix = np.array([1500000000000,1500000000048,1500000000075,1500000000100,1500000000150])
    >>> df['unixtime'] = unix
    >>> print(df)
            unixtime   A   B   C
    0  1500000000000   1   2   3
    1  1500000000048   5   6   7
    2  1500000000075   9  10  11
    3  1500000000100  13  14  15
    4  1500000000150  17  18  19
    >>> newSamplingRate = 20
    >>> newDf = resample(df, timeColHeader, newSamplingRate)
    >>> print(newDf)
            unixtime          A          B          C
    0  1500000000000   1.000000   2.000000   3.000000
    1  1500000000050   5.296295   6.296295   7.296295
    2  1500000000100  13.000000  14.000000  15.000000
    3  1500000000150  17.000000  18.000000  19.000000

    >>> newSamplingRate = 33
    >>> newDf = resample(df, timeColHeader, newSamplingRate)
    >>> print(newDf)
            unixtime          A          B          C
    0  1500000000000   1.000000   2.000000   3.000000
    1  1500000000030   3.525238   4.525238   5.525238
    2  1500000000060   6.867554   7.867554   8.867554
    3  1500000000090  11.545441  12.545441  13.545441
    4  1500000000121  14.696960  15.696960  16.696960

    (Note: the 5th unixtime is 1500000000121 instead of 1500000000120, since 5th sampling is 121.21212121ms after 1st sampling.
    
    development log:
    1.
    # always take the first timestamp time[0]
    # if starttime == None:
    newSignalList = [signalArr[0]]
    newUnixtimeList = [unixtimeArr[0]]
    # else:
    #     newUnixtimeList = [starttime]
        # if starttime >= signalArr[0]
        # newSignalList = interpolate(unixtimeArr[tIndAfter-1], signalArr[tIndAfter-1], unixtimeArr[tIndAfter], signalArr[tIndAfter], t)
    
    2.
    # if gapTolerance == 0 or \
    #     ((abs(unixtimeArr[tIndAfter-1]-t)<=gapTolerance) and \
    #     (abs(unixtimeArr[tIndAfter]-t)<=gapTolerance)):

    if gapTolerance == 0 or \
        (abs(unixtimeArr[tIndAfter-1]-unixtimeArr[tIndAfter])<=gapTolerance):

    -----
    """

    originalNameOrder = list(dataDf.columns.values)

    unixtimeArr = dataDf[timeColHeader].values
    deltaT = 1000.0/samplingRate
    
    dataDf = dataDf.drop(timeColHeader, axis=1)
    dataArr = dataDf.values
    names = list(dataDf.columns.values)

    n = len(unixtimeArr)
    newDataList = []
    
    if n<2:
        return

    if fixedTimeColumn is None:
        #Looping through columns to apply the resampling method for each column
        for c in range(dataArr.shape[1]):
            signalArr = dataArr[:,c]

            # always take the first timestamp time[0]
            newSignalList = [signalArr[0]]
            newUnixtimeList = [unixtimeArr[0]]

            t = unixtimeArr[0] + deltaT
            tIndAfter = 1

            # iterate through the original signal
            while True:
                # look for the interval that contains 't'
                i0 = tIndAfter
                for i in range(i0,n):# name indBefore/after
                    if  t <= unixtimeArr[i]:#we found the needed time index
                        tIndAfter = i
                        break

                # interpolate in the right interval, gapTolenance=0 means inf tol,
                if gapTolerance == 0 or \
                    (abs(unixtimeArr[tIndAfter-1]-unixtimeArr[tIndAfter])<=gapTolerance):
                    s = interpolate(unixtimeArr[tIndAfter-1], signalArr[tIndAfter-1], \
                                    unixtimeArr[tIndAfter], signalArr[tIndAfter], t)
                else:
                    s = np.nan

                # apppend the new interpolated sample to the new signal and update the new time vector
                newSignalList.append(s)
                newUnixtimeList.append(int(t))
                # take step further on time
                t = t + deltaT
                # check the stop condition
                if t>unixtimeArr[-1]:
                    break

            newDataList.append(newSignalList)
            newDataArr = np.transpose(np.array(newDataList))

        dataDf = pd.DataFrame(data = newDataArr, columns = names)
        dataDf[timeColHeader] = np.array(newUnixtimeList)

        # change to the original column order
        dataDf = dataDf[originalNameOrder]

    else:  #if fixedTimeColumn not None:
        #Looping through columns to apply the resampling method for each column
        for c in range(dataArr.shape[1]):
            signalArr = dataArr[:,c]
            newSignalList = []
            newUnixtimeList = []

            iFixedTime = 0

            t = fixedTimeColumn[iFixedTime]
            tIndAfter = 0
            outOfRange = 1

            # iterate through the original signal
            while True:
                # look for the interval that contains 't'
                i0 = tIndAfter
                for i in range(i0,n):
                    if  t <= unixtimeArr[i]:#we found the needed time index
                        tIndAfter = i
                        outOfRange = 0
                        break

                if outOfRange:
                    s = np.nan
                else:
                    # interpolate in the right interval
                    if tIndAfter == 0: # means unixtimeArr[0] > t, there is no element smaller than t
                        s = np.nan
                    elif gapTolerance == 0 or \
                        (abs(unixtimeArr[tIndAfter-1] - unixtimeArr[tIndAfter]) <= gapTolerance):
                        s = interpolate(unixtimeArr[tIndAfter-1], signalArr[tIndAfter-1], \
                                        unixtimeArr[tIndAfter], signalArr[tIndAfter], t)
                    else:
                        s = np.nan

                # apppend the new interpolated sample to the new signal and update the new time vector
                newSignalList.append(s)
                newUnixtimeList.append(int(t))

                # check the stop condition
                if t>unixtimeArr[-1]:
                    break
                # take step further on time
                iFixedTime += 1

                if iFixedTime>=len(fixedTimeColumn):
                    break
                t = fixedTimeColumn[iFixedTime]

            newDataList.append(newSignalList)
            newDataArr = np.transpose(np.array(newDataList))

        dataDf = pd.DataFrame(data = newDataArr, columns = names)
        dataDf[timeColHeader] = np.array(newUnixtimeList)

        # change to the original column order
        dataDf = dataDf[originalNameOrder]
    return dataDf

def interpolate(t1, s1, t2, s2, t):
    """Interpolates at parameter 't' between points (t1,s1) and (t2,s2)
    """

    if(t1<=t and t<=t2): #we check if 't' is out of bounds (between t1 and t2)
        m = float(s2 - s1)/(t2 - t1)
        b = s1 - m*t1
        return m*t + b
    else:
        return np.nan




def main():
    # path of study folder
    study_path = str(sys.argv[1])
    # participants# (eg. "P301 P302 P401")
    p_nums = str(sys.argv[2])

    model = XGBClassifier(learning_rate=0.01,
                        n_estimators=400,           # 树的个数-10棵树建立xgboost
                        max_depth=10,               # 树的深度
                        min_child_weight = 1,      # 叶子节点最小权重
                        gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                        subsample=1,               # 所有样本建立决策树
                        colsample_btree=1,         # 所有特征建立决策树
                        scale_pos_weight=1,        # 解决样本个数不平衡的问题
                        random_state=7,           # 随机数
                        slient = 0,
                        nthread = 4
                        )

    model = joblib.load(study_path+'/xgbc.dat')


    participants = p_nums.split(' ')

    




if __name__ == '__main__':
    main()