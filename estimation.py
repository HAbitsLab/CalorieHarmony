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


def test_and_estimate(study_path,p_nums):
    """
    The classification script needs to be run first to have the model built.
    This function loads the model and test the performance using the input participants' data.
    It estimates the results and plots the comparison graphs.
    
    Parameters:
        Required:
        - study_path -- the path of the study folder
        - p_nums -- participant numbers separated by space (eg. "P301 P302 P401")
    """

    model = XGBClassifier(learning_rate = 0.01,
                        n_estimators = 400,
                        max_depth = 10,
                        min_child_weight = 1,
                        gamma = 0,
                        subsample = 1,
                        colsample_btree = 1,
                        scale_pos_weight = 1,
                        random_state = 7,
                        slient = 0,
                        nthread = 4
                        )

    model = joblib.load(study_path+'/xgbc.dat')


    participants = p_nums.split(' ') 


    t0 = time()

    data_gyro = []
    target = []

    tables = []
    for p in participants:
        path_ts = study_path+'/'+p+'/In Lab/'+p+' In Lab Log.csv'
        df_ts = pd.read_csv(path_ts, index_col=None, header=0)
        
        path_gyro = study_path+'/'+p+'/In Lab/Wrist/Aggregated/Gyroscope/Gyroscope_resampled.csv'
        if not os.path.exists(path_gyro):
            df_gyro_raw = pd.read_csv(study_path+'/'+p+'/In Lab/Wrist/Aggregated/Gyroscope/Gyroscope.csv', index_col=None, header=0)
            df_gyro = resample(df_gyro_raw, 'Time', 20, 100)
            df_gyro.to_csv(path_gyro, index=False)
        df_gyro = pd.read_csv(path_gyro, index_col=None, header=0)
        df_gyro['Datetime'] = pd.to_datetime(df_gyro['Time'], unit='ms', utc=True).dt.tz_convert('America/Chicago').dt.tz_localize(None)
    
        sedentary_activities = ['breathing','computer','standing','reading','lie down']

        path_table = study_path+'/'+p+'/In Lab/Summary/Actigraph/'+p+' In Lab IntensityMETActivityLevel.csv'
        df_table = pd.read_csv(path_table, index_col=None, header=0)
        
        prediction = []
        
        for i in range(len(df_ts['Activity'])):
            if not pd.isnull(df_ts['Start Time'][i]):
                st = pd.to_datetime(df_ts['Start Time'][i], format='%H:%M:%S')
                et = pd.to_datetime(df_ts['End Time'][i], format='%H:%M:%S')
                start_time = pd.Timestamp.combine(pd.to_datetime(df_ts['Start Date'][i], format='%m/%d/%y').date(), st.time())
                for j in range(int((et-st).seconds/60)):
                    end_time = start_time + pd.DateOffset(minutes= 1)
                    temp_gyro = df_gyro.loc[(df_gyro['Datetime'] >= start_time) & (df_gyro['Datetime'] < end_time)].reset_index(drop=True)
                    
                    if df_ts['Activity'][i] in sedentary_activities:
                        target.append(0)
                        this_min_gyro = [temp_gyro['rotX'],temp_gyro['rotY'],temp_gyro['rotZ']]
                        data_gyro.append(this_min_gyro)
                    
                    if df_ts['Activity'][i] not in sedentary_activities:
                        target.append(1)
                        this_min_gyro = [temp_gyro['rotX'],temp_gyro['rotY'],temp_gyro['rotZ']]
                        data_gyro.append(this_min_gyro)

                    if len(temp_gyro['rotX'])!=0:
                        this_min_gyro = [temp_gyro['rotX'],temp_gyro['rotY'],temp_gyro['rotZ']]
                        if np.count_nonzero(np.isnan(this_min_gyro[0]))>(this_min_gyro[0].size/2):
                            prediction.append(-1)
                        else:
                            this_min_gyro_new = []
                            this_min_gyro_new.append(this_min_gyro[0]+this_min_gyro[1]+this_min_gyro[2])
                            this_min_gyro_new = np.array(this_min_gyro_new)

                            model_output = model.predict(this_min_gyro_new)
                            prediction.append(model_output[0])
                                
                    if len(temp_gyro['rotX'])==0:
                        prediction.append(-1)
                            
                    start_time += pd.DateOffset(minutes= 1)
            
        df_table['model_classification'] = prediction
        tables.append(df_table)
        
        
    t1 = time()
    print("Preprocessing time (minutes): %.4g" % (float(t1 - t0)/float(60)))

    new_data_gyro = [n for n in data_gyro if np.count_nonzero(np.isnan(n[0]))<(n[0].size/2)]
    new_target_gyro = [target[i] for i in range(len(data_gyro)) if np.count_nonzero(np.isnan(data_gyro[i][0]))<(data_gyro[i][0].size/2)]

    np_data_gyro = np.array(new_data_gyro)
    np_target_gyro = np.array(new_target_gyro)

    np_data_gyro_new = []
    for i in range(len(np_data_gyro)):
        data_i = np.array(np_data_gyro[i])
        np_data_gyro_new.append(data_i[0]+data_i[1]+data_i[2])
    np_data_gyro_new = np.array(np_data_gyro_new)

    y_pred = model.predict(np_data_gyro_new)
    print("Test Accuracy: %.4g" % metrics.accuracy_score(np_target_gyro, y_pred))

    # rescale the intensity values
    # add the results to the tables

    # intensity_coef = 8/13.7302
    intensity_coef = 0.4062

    for t in tables:
        t['scaled_intensity'] = t['Watch Intensity']*intensity_coef + 1.3


    # adjust the rescaled intensity values according to the classification results
    # add to the tables

    for t in tables:
        estimation = []
        for i in range(len(t['model_classification'])):
            c = t['model_classification'][i]
            s = t['scaled_intensity'][i]
            
            if c==-1:
                if s<1.3:
                    estimation.append(1.3)
                else:
                    estimation.append(s)
            elif c==0:
                if s<1.3:
                    estimation.append(1.3)
                elif s>1.5:
                    estimation.append(1.5)
                else:
                    estimation.append(s)
            elif c==1:
                if s<1.5:
                    estimation.append(1.5)
                elif s>10:
                    estimation.append(10)
                else:
                    estimation.append(s)
                    
        t['estimation'] = estimation


    # take the values from the tables to vitualize

    df_table_all = pd.concat(tables).reset_index(drop=True)

    l_vm3_all = df_table_all['MET (VM3)'].tolist()
    l_estimation_all = df_table_all['estimation'].tolist()
    l_ainsworth = df_table_all['MET (Ainsworth)'].tolist()
    l_scaled_intensity = df_table_all['scaled_intensity'].tolist()
    l_google_fit = df_table_all['Google Fit'].tolist()

    l_google_fit = [l_google_fit[i] for i in range(len(l_scaled_intensity)) if not np.isnan(l_scaled_intensity[i])]
    l_vm3_all = [l_vm3_all[i] for i in range(len(l_scaled_intensity)) if not np.isnan(l_scaled_intensity[i])]
    l_estimation_all = [l_estimation_all[i] for i in range(len(l_scaled_intensity)) if not np.isnan(l_scaled_intensity[i])]
    l_ainsworth = [l_ainsworth[i] for i in range(len(l_scaled_intensity)) if not np.isnan(l_scaled_intensity[i])]
    l_scaled_intensity = [l_scaled_intensity[i] for i in range(len(l_scaled_intensity)) if not np.isnan(l_scaled_intensity[i])]

    vm3_all_reshaped = np.array(l_vm3_all).reshape(-1, 1)
    estimation_all_reshaped = np.array(l_estimation_all).reshape(-1, 1)
    ainsworth_all_reshaped = np.array(l_ainsworth).reshape(-1, 1)
    google_fit_reshaped = np.array(l_google_fit).reshape(-1, 1)


    act_dict = {}
    activities = ['breathing','computer','slow walk','fast walk','standing','squats','reading','aerobics','sweeping','pushups','running','lie down','stairs']
    for a in activities:
        act_dict[a] = [[],[]]
    for i in range(len(df_table_all['Activity'])):
        act_dict[df_table_all['Activity'][i]][0].append(df_table_all['estimation'][i])
        act_dict[df_table_all['Activity'][i]][1].append(df_table_all['MET (Ainsworth)'][i])
        
        
    regr = linear_model.LinearRegression()

    regr.fit(estimation_all_reshaped, ainsworth_all_reshaped)

    y_pred = regr.predict(estimation_all_reshaped)


    fig = go.Figure()

    for a in act_dict:
        fig.add_trace(go.Scatter(x=act_dict[a][0], y=act_dict[a][1], mode='markers', name=a))

    y_plot = np.reshape(y_pred,y_pred.shape[0])
    fig.add_trace(go.Scatter(x=l_estimation_all, y=y_plot, mode='lines', name='linear regression', line=dict(color='red', width=4)))

    fig.update_layout(title='Linear Regression',
                    xaxis_title='Estimation',
                    yaxis_title='Ainsworth METs')

    # fig.show()
    py.offline.plot(fig, filename=study_path+'/estimation.html')

    print("The r2 score for our estimation is: %.4g" % (r2_score(ainsworth_all_reshaped, y_pred)))


    act_dict = {}
    activities = ['breathing','computer','slow walk','fast walk','standing','squats','reading','aerobics','sweeping','pushups','running','lie down','stairs']
    for a in activities:
        act_dict[a] = [[],[]]
    for i in range(len(df_table_all['Activity'])):
        act_dict[df_table_all['Activity'][i]][0].append(df_table_all['MET (VM3)'][i])
        act_dict[df_table_all['Activity'][i]][1].append(df_table_all['MET (Ainsworth)'][i])
        
        
    regr = linear_model.LinearRegression()

    regr.fit(vm3_all_reshaped, ainsworth_all_reshaped)

    y_pred = regr.predict(vm3_all_reshaped)


    fig = go.Figure()

    for a in act_dict:
        fig.add_trace(go.Scatter(x=act_dict[a][0], y=act_dict[a][1], mode='markers', name=a))

    y_plot = np.reshape(y_pred,y_pred.shape[0])
    fig.add_trace(go.Scatter(x=l_vm3_all, y=y_plot, mode='lines', name='linear regression', line=dict(color='red', width=4)))

    fig.update_layout(title='Linear Regression',
                    xaxis_title='VM3 METs',
                    yaxis_title='Ainsworth METs')

    # fig.show()
    py.offline.plot(fig, filename=study_path+'/actigraphVM3.html')

    print("The r2 score for ActiGraph VM3 is: %.4g" % (r2_score(ainsworth_all_reshaped, y_pred)))


    act_dict = {}
    activities = ['breathing','computer','slow walk','fast walk','standing','squats','reading','aerobics','sweeping','pushups','running','lie down','stairs']
    for a in activities:
        act_dict[a] = [[],[]]
    for i in range(len(df_table_all['Activity'])):
        act_dict[df_table_all['Activity'][i]][0].append(df_table_all['Google Fit'][i])
        act_dict[df_table_all['Activity'][i]][1].append(df_table_all['MET (Ainsworth)'][i])
        
        
    regr = linear_model.LinearRegression()

    regr.fit(google_fit_reshaped, ainsworth_all_reshaped)

    y_pred = regr.predict(google_fit_reshaped)


    fig = go.Figure()

    for a in act_dict:
        fig.add_trace(go.Scatter(x=act_dict[a][0], y=act_dict[a][1], mode='markers', name=a))

    y_plot = np.reshape(y_pred,y_pred.shape[0])
    fig.add_trace(go.Scatter(x=l_google_fit, y=y_plot, mode='lines', name='linear regression', line=dict(color='red', width=4)))

    fig.update_layout(title='Linear Regression',
                    xaxis_title='Google Fit Calorie Reading',
                    yaxis_title='Ainsworth METs')

    # fig.show()
    py.offline.plot(fig, filename=study_path+'/googleFit.html')

    print("The r2 score for Google Fit is: %.4g" % (r2_score(ainsworth_all_reshaped, y_pred)))


def main():
    # path of study folder
    study_path = str(sys.argv[1])
    # participants# (eg. "P301 P302 P401")
    p_nums = str(sys.argv[2])

    test_and_estimate(study_path,p_nums)



if __name__ == '__main__':
    main()