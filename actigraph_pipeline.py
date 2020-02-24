import numpy as np
import pandas as pd
import glob
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from plotly import graph_objs as go
import inspect
import shutil
from acti_watch_summary import generate_summary


def check_split_weartime_file(study_path, p_num, state):
    summary_files = ['SummaryDayLevel.csv', 'SummaryWearTime.csv']
    path_to_acti_summary = os.path.join(study_path, p_num, state, 'Summary', 'Actigraph')

    sum_files_exists = [sum_file for sum_file in summary_files if
                        os.path.isfile(os.path.join(path_to_acti_summary, sum_file))]

    if len(summary_files) != len(sum_files_exists):
        # weartime_file = os.path.join(study_path, p_num, state, 'Actigraph', f'P{subj_ID} {state} Weartime.csv')
        weartime_file = os.path.join(study_path, p_num, state, 'Actigraph', 'Clean', f'{p_num} {state} Weartime.csv')
        weartime_df = pd.read_csv(weartime_file, skiprows=1)
        split_ind = weartime_df[weartime_df['Vector Magnitude'].isnull()].index[0]

        weartime_df1 = weartime_df[:split_ind]

        weartime_df2 = weartime_df[split_ind:]
        weartime_df2.columns = weartime_df2.iloc[0]
        weartime_df2 = weartime_df2.drop(weartime_df2.index[0])
        weartime_df2 = weartime_df2.dropna(axis=1, how='all')
        weartime_df2 = weartime_df2.reset_index(drop=True)

        if not os.path.exists(path_to_acti_summary):
            os.makedirs(path_to_acti_summary)

        weartime_df1.to_csv(os.path.join(path_to_acti_summary, 'SummaryDayLevel.csv'), index=False)
        weartime_df2.to_csv(os.path.join(path_to_acti_summary, 'SummaryWearTime.csv'), index=False)
        print('Split weartime file into SummaryDayLevel and SummaryWearTime')


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
    deltaT = 1000.0 / samplingRate

    dataDf = dataDf.drop(timeColHeader, axis=1)
    dataArr = dataDf.values
    names = list(dataDf.columns.values)

    n = len(unixtimeArr)
    newDataList = []

    if n < 2:
        return

    if fixedTimeColumn is None:
        # Looping through columns to apply the resampling method for each column
        for c in range(dataArr.shape[1]):
            signalArr = dataArr[:, c]

            # always take the first timestamp time[0]
            newSignalList = [signalArr[0]]
            newUnixtimeList = [unixtimeArr[0]]

            t = unixtimeArr[0] + deltaT
            tIndAfter = 1

            # iterate through the original signal
            while True:
                # look for the interval that contains 't'
                i0 = tIndAfter
                for i in range(i0, n):  # name indBefore/after
                    if t <= unixtimeArr[i]:  # we found the needed time index
                        tIndAfter = i
                        break

                # interpolate in the right interval, gapTolenance=0 means inf tol,
                if gapTolerance == 0 or \
                        (abs(unixtimeArr[tIndAfter - 1] - unixtimeArr[tIndAfter]) <= gapTolerance):
                    s = interpolate(unixtimeArr[tIndAfter - 1], signalArr[tIndAfter - 1], \
                                    unixtimeArr[tIndAfter], signalArr[tIndAfter], t)
                else:
                    s = np.nan

                # apppend the new interpolated sample to the new signal and update the new time vector
                newSignalList.append(s)
                newUnixtimeList.append(int(t))
                # take step further on time
                t = t + deltaT
                # check the stop condition
                if t > unixtimeArr[-1]:
                    break

            newDataList.append(newSignalList)
            newDataArr = np.transpose(np.array(newDataList))

        dataDf = pd.DataFrame(data=newDataArr, columns=names)
        dataDf[timeColHeader] = np.array(newUnixtimeList)

        # change to the original column order
        dataDf = dataDf[originalNameOrder]

    else:  # if fixedTimeColumn not None:
        # Looping through columns to apply the resampling method for each column
        for c in range(dataArr.shape[1]):
            signalArr = dataArr[:, c]
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
                for i in range(i0, n):
                    if t <= unixtimeArr[i]:  # we found the needed time index
                        tIndAfter = i
                        outOfRange = 0
                        break

                if outOfRange:
                    s = np.nan
                else:
                    # interpolate in the right interval
                    if tIndAfter == 0:  # means unixtimeArr[0] > t, there is no element smaller than t
                        s = np.nan
                    elif gapTolerance == 0 or \
                            (abs(unixtimeArr[tIndAfter - 1] - unixtimeArr[tIndAfter]) <= gapTolerance):
                        s = interpolate(unixtimeArr[tIndAfter - 1], signalArr[tIndAfter - 1], \
                                        unixtimeArr[tIndAfter], signalArr[tIndAfter], t)
                    else:
                        s = np.nan

                # apppend the new interpolated sample to the new signal and update the new time vector
                newSignalList.append(s)
                newUnixtimeList.append(int(t))

                # check the stop condition
                if t > unixtimeArr[-1]:
                    break
                # take step further on time
                iFixedTime += 1

                if iFixedTime >= len(fixedTimeColumn):
                    break
                t = fixedTimeColumn[iFixedTime]

            newDataList.append(newSignalList)
            newDataArr = np.transpose(np.array(newDataList))

        dataDf = pd.DataFrame(data=newDataArr, columns=names)
        dataDf[timeColHeader] = np.array(newUnixtimeList)

        # change to the original column order
        dataDf = dataDf[originalNameOrder]
    return dataDf


def interpolate(t1, s1, t2, s2, t):
    """Interpolates at parameter 't' between points (t1,s1) and (t2,s2)
    """

    if (t1 <= t and t <= t2):  # we check if 't' is out of bounds (between t1 and t2)
        m = float(s2 - s1) / (t2 - t1)
        b = s1 - m * t1
        return m * t + b
    else:
        return np.nan


def watch_add_datetime(watch_df):
    watch_df['Datetime'] = pd.to_datetime(watch_df['Time'], unit='ms', utc=True).dt.tz_convert(
        'America/Chicago').dt.tz_localize(None)


def actigraph_add_datetime(actigraph_data):
    datetime = []
    for i in range(len(actigraph_data['date'])):
        date = pd.to_datetime(actigraph_data['date'][i], format='%m/%d/%Y').date()
        time = pd.to_datetime(actigraph_data['epoch'][i], format='%I:%M:%S %p').time()
        temp = pd.Timestamp.combine(date, time)
        datetime.append(temp)
    actigraph_data['Datetime'] = datetime


def plot_data_watch(df_watch, df_ts, study_path, p_num, state):
    ts_start_lab = pd.to_datetime(df_ts['Trial Start'][0], format='%m/%d/%y %H:%M:%S')
    ts_end_lab = pd.to_datetime(df_ts['Trial End'][0], format='%m/%d/%y %H:%M:%S')
    df_watch_lab = df_watch.loc[
        (df_watch['Datetime'] >= ts_start_lab) & (df_watch['Datetime'] <= ts_end_lab)].reset_index(drop=True)
    df_watch_lab_20 = df_watch_lab.loc[df_watch_lab.index % 20 == 0].reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df_watch_lab_20['Datetime'], y=df_watch_lab_20['accX'], mode='lines+markers', name='x-axis'))
    fig.add_trace(
        go.Scatter(x=df_watch_lab_20['Datetime'], y=df_watch_lab_20['accY'], mode='lines+markers', name='y-axis'))
    fig.add_trace(
        go.Scatter(x=df_watch_lab_20['Datetime'], y=df_watch_lab_20['accZ'], mode='lines+markers', name='z-axis'))

    show_legend_b = True
    for i in range(int(len(df_ts['Date']))):
        ts_start_lab_i = pd.Timestamp.combine(pd.to_datetime(df_ts['Start Date'][i], format='%m/%d/%y').date(),
                                              pd.to_datetime(df_ts['Start Time'][i], format='%H:%M:%S').time())
        ts_end_lab_i = pd.Timestamp.combine(pd.to_datetime(df_ts['End Date'][i], format='%m/%d/%y').date(),
                                            pd.to_datetime(df_ts['End Time'][i], format='%H:%M:%S').time())
        df_watch_lab_i = df_watch_lab_20.loc[(df_watch_lab_20['Datetime'] >= ts_start_lab_i) & (
                    df_watch_lab_20['Datetime'] <= ts_end_lab_i)].reset_index(drop=True)
        if df_ts['Activity'][i] == 'breathing':
            if show_legend_b:
                fig.add_trace(go.Scatter(x=df_watch_lab_i['Datetime'], y=[-25 for n in df_watch_lab_i['Datetime']],
                                         name=df_ts['Activity'][i], line=dict(color='white', width=10)))
                show_legend_b = False
            else:
                fig.add_trace(go.Scatter(x=df_watch_lab_i['Datetime'], y=[-25 for n in df_watch_lab_i['Datetime']],
                                         showlegend=False, line=dict(color='white', width=10)))
        else:
            fig.add_trace(go.Scatter(x=df_watch_lab_i['Datetime'], y=[-25 for n in df_watch_lab_i['Datetime']],
                                     name=df_ts['Activity'][i], line=dict(width=10)))

    show_legend_r = True
    for i in range(int(len(df_ts['Start Date']) - 1)):
        rest_start = pd.Timestamp.combine(pd.to_datetime(df_ts['End Date'][i], format='%m/%d/%y').date(),
                                          pd.to_datetime(df_ts['End Time'][i], format='%H:%M:%S').time())
        rest_end = pd.Timestamp.combine(pd.to_datetime(df_ts['Start Date'][i + 1], format='%m/%d/%y').date(),
                                        pd.to_datetime(df_ts['Start Time'][i + 1], format='%H:%M:%S').time())
        df_watch_lab_i = df_watch_lab_20.loc[
            (df_watch_lab_20['Datetime'] > rest_start) & (df_watch_lab_20['Datetime'] < rest_end)].reset_index(
            drop=True)
        if show_legend_r:
            fig.add_trace(
                go.Scatter(x=df_watch_lab_i['Datetime'], y=[-25 for n in df_watch_lab_i['Datetime']], name='rest',
                           line=dict(color='black', width=10)))
            show_legend_r = False
        else:
            fig.add_trace(
                go.Scatter(x=df_watch_lab_i['Datetime'], y=[-25 for n in df_watch_lab_i['Datetime']], showlegend=False,
                           line=dict(color='black', width=10)))

    fig.write_image(study_path + "/" + p_num + "/" + state + "/Summary/" + p_num + ' ' + state + ' watch.png')
    # fig.show()


def plot_data_acti(df_acti, df_ts, study_path, p_num, state):
    ts_start_lab = pd.to_datetime(df_ts['Trial Start'][0], format='%m/%d/%y %H:%M:%S')
    ts_end_lab = pd.to_datetime(df_ts['Trial End'][0], format='%m/%d/%y %H:%M:%S')
    df_acti_lab = df_acti.loc[(df_acti['Datetime'] >= ts_start_lab) & (df_acti['Datetime'] <= ts_end_lab)].reset_index(
        drop=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_acti_lab['Datetime'], y=df_acti_lab['axis2'], mode='lines+markers', name='x-axis'))
    fig.add_trace(go.Scatter(x=df_acti_lab['Datetime'], y=df_acti_lab['axis1'], mode='lines+markers', name='y-axis'))
    fig.add_trace(go.Scatter(x=df_acti_lab['Datetime'], y=df_acti_lab['axis3'], mode='lines+markers', name='z-axis'))

    show_legend_b = True
    for i in range(int(len(df_ts['Start Date']))):
        ts_start_lab_i = pd.Timestamp.combine(pd.to_datetime(df_ts['Start Date'][i], format='%m/%d/%y').date(),
                                              pd.to_datetime(df_ts['Start Time'][i], format='%H:%M:%S').time())
        ts_end_lab_i = pd.Timestamp.combine(pd.to_datetime(df_ts['End Date'][i], format='%m/%d/%y').date(),
                                            pd.to_datetime(df_ts['End Time'][i], format='%H:%M:%S').time())
        df_acti_lab_i = df_acti_lab.loc[
            (df_acti_lab['Datetime'] >= ts_start_lab_i) & (df_acti_lab['Datetime'] <= ts_end_lab_i)].reset_index(
            drop=True)
        if df_ts['Activity'][i] == 'breathing':
            if show_legend_b:
                fig.add_trace(go.Scatter(x=df_acti_lab_i['Datetime'], y=[-500 for n in df_acti_lab_i['Datetime']],
                                         name=df_ts['Activity'][i], line=dict(color='white', width=10)))
                show_legend_b = False
            else:
                fig.add_trace(go.Scatter(x=df_acti_lab_i['Datetime'], y=[-500 for n in df_acti_lab_i['Datetime']],
                                         showlegend=False, line=dict(color='white', width=10)))
        else:
            fig.add_trace(go.Scatter(x=df_acti_lab_i['Datetime'], y=[-500 for n in df_acti_lab_i['Datetime']],
                                     name=df_ts['Activity'][i], line=dict(width=10)))

    show_legend_r = True
    for i in range(int(len(df_ts['Date']) - 1)):
        rest_start = pd.Timestamp.combine(pd.to_datetime(df_ts['End Date'][i], format='%m/%d/%y').date(),
                                          pd.to_datetime(df_ts['End Time'][i], format='%H:%M:%S').time())
        rest_end = pd.Timestamp.combine(pd.to_datetime(df_ts['Start Date'][i + 1], format='%m/%d/%y').date(),
                                        pd.to_datetime(df_ts['Start Time'][i + 1], format='%H:%M:%S').time())
        df_acti_lab_i = df_acti_lab.loc[
            (df_acti_lab['Datetime'] > rest_start) & (df_acti_lab['Datetime'] < rest_end)].reset_index(drop=True)
        if show_legend_r:
            fig.add_trace(
                go.Scatter(x=df_acti_lab_i['Datetime'], y=[-500 for n in df_acti_lab_i['Datetime']], name='rest',
                           line=dict(color='black', width=10)))
            show_legend_r = False
        else:
            fig.add_trace(
                go.Scatter(x=df_acti_lab_i['Datetime'], y=[-500 for n in df_acti_lab_i['Datetime']], showlegend=False,
                           line=dict(color='black', width=10)))

    fig.write_image(study_path + "/" + p_num + "/" + state + "/Summary/" + p_num + ' ' + state + ' actigraph.png')
    # fig.show()


def get_intensity(watch_df, st):
    et = st + pd.DateOffset(minutes=1)
    temp = watch_df.loc[(watch_df['Datetime'] >= st) & (watch_df['Datetime'] < et)].reset_index(drop=True)
    sum_x_sq = 0
    sum_y_sq = 0
    sum_z_sq = 0
    sum_x = 0
    sum_y = 0
    sum_z = 0
    count = 0
    for i in range(0, len(temp)):
        if not np.isnan(temp['accX'][i]):
            sum_x_sq += temp['accX'][i] ** 2
            sum_y_sq += temp['accY'][i] ** 2
            sum_z_sq += temp['accZ'][i] ** 2
            sum_x += temp['accX'][i]
            sum_y += temp['accY'][i]
            sum_z += temp['accZ'][i]
            count += 1
    if count != 0:
        Q = sum_x_sq + sum_y_sq + sum_z_sq
        P = sum_x ** 2 + sum_y ** 2 + sum_z ** 2
        K = ((Q - P / count) / (count - 1)) ** 0.5
        return K
    else:
        return np.nan


def get_met_freedson(df_acti, st):
    et = st + pd.DateOffset(minutes=1)
    temp = df_acti.loc[(df_acti['Datetime'] >= st) & (df_acti['Datetime'] < et)].reset_index(drop=True)
    if len(temp['axis1']) > 0:
        met = temp['axis1'][0] * 0.000795 + 1.439008
        return met
    else:
        return np.nan


def get_met_vm3(df_acti, st):
    et = st + pd.DateOffset(minutes=1)
    temp = df_acti.loc[(df_acti['Datetime'] >= st) & (df_acti['Datetime'] < et)].reset_index(drop=True)
    vm3 = (temp['axis1'][0] ** 2 + temp['axis2'][0] ** 2 + temp['axis3'][0] ** 2) ** 0.5
    met = 0.000863 * vm3 + 0.668876
    return met

def get_met_output(df_acti, st):
    et = st + pd.DateOffset(minutes=1)
    temp = df_acti.loc[(df_acti['Datetime'] >= st) & (df_acti['Datetime'] < et)].reset_index(drop=True)
    output = temp['MET rate'][0]
    return output


def plot_data_watch_without_ts(df_watch, study_path, p_num, state):
    df_watch_20 = df_watch.loc[df_watch.index % 20 == 0].reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_watch_20['Datetime'], y=df_watch_20['accX'], mode='lines', name='x-axis'))
    fig.add_trace(go.Scatter(x=df_watch_20['Datetime'], y=df_watch_20['accY'], mode='lines', name='y-axis'))
    fig.add_trace(go.Scatter(x=df_watch_20['Datetime'], y=df_watch_20['accZ'], mode='lines', name='z-axis'))

    fig.write_image(study_path + "/" + p_num + "/" + state + "/Actigraph/" + p_num + ' ' + state + ' watch.png')
    # fig.show()


def plot_data_acti_without_ts(df_acti, study_path, p_num, state):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_acti['Datetime'], y=df_acti['axis2'], mode='lines', name='x-axis'))
    fig.add_trace(go.Scatter(x=df_acti['Datetime'], y=df_acti['axis1'], mode='lines', name='y-axis'))
    fig.add_trace(go.Scatter(x=df_acti['Datetime'], y=df_acti['axis3'], mode='lines', name='z-axis'))

    fig.write_image(study_path + "/" + p_num + "/" + state + "/Actigraph/" + p_num + ' ' + state + ' actigraph.png')
    # fig.show()


def get_3d_sqd(watch_df, st):
    et = st + pd.DateOffset(minutes=1)
    temp = watch_df.loc[(watch_df['Datetime'] >= st) & (watch_df['Datetime'] < et)].reset_index(drop=True)
    sum_x_sq_diff = 0
    sum_y_sq_diff = 0
    sum_z_sq_diff = 0
    count = 0
    for i in range(0, len(temp)):
        if i >= 50:
            if not np.isnan(temp['accX'][i]):
                val1 = temp['accX'][i] - temp['accX'][i - 50]
                val2 = temp['accY'][i] - temp['accY'][i - 50]
                val3 = temp['accZ'][i] - temp['accZ'][i - 50]
                # if abs(temp['accX'][i])>1:
                sum_x_sq_diff += val1 ** 2
                # if abs(temp['accY'][i])>1:
                sum_y_sq_diff += val2 ** 2
                # if abs(temp['accZ'][i]-9)>1:
                sum_z_sq_diff += val3 ** 2
                count += 1
    if count != 0:
        return [sum_x_sq_diff / count, sum_y_sq_diff / count, sum_z_sq_diff / count]
    else:
        return [np.nan, np.nan, np.nan]


def get_3d_raw(watch_df, st):
    et = st + pd.DateOffset(minutes=1)
    temp = watch_df.loc[(watch_df['Datetime'] >= st) & (watch_df['Datetime'] < et)].reset_index(drop=True)
    x_mean = temp['accX'].mean()
    y_mean = temp['accY'].mean()
    z_mean = temp['accZ'].mean()
    if len(temp['accX']) != 0:
        return [x_mean, y_mean, z_mean]
    else:
        return [np.nan, np.nan, np.nan]


def get_sign_change_count(watch_df, st):
    et = st + pd.DateOffset(minutes=1)
    temp = watch_df.loc[(watch_df['Datetime'] >= st) & (watch_df['Datetime'] < et)].reset_index(drop=True)
    temp1 = temp[:-1]
    temp2 = temp[1:]

    temp1_sign_x = [n >= 0 for n in temp1['accX']]
    temp2_sign_x = [n >= 0 for n in temp2['accX']]
    compare_x = [0 if temp1_sign_x[i] == temp2_sign_x[i] else 1 for i in range(len(temp1_sign_x))]
    change_x = compare_x.count(1)

    temp1_sign_y = [n >= 0 for n in temp1['accY']]
    temp2_sign_y = [n >= 0 for n in temp2['accY']]
    compare_y = [0 if temp1_sign_y[i] == temp2_sign_y[i] else 1 for i in range(len(temp1_sign_y))]
    change_y = compare_y.count(1)

    temp1_sign_z = [n >= 0 for n in temp1['accZ']]
    temp2_sign_z = [n >= 0 for n in temp2['accZ']]
    compare_z = [0 if temp1_sign_z[i] == temp2_sign_z[i] else 1 for i in range(len(temp1_sign_z))]
    change_z = compare_z.count(1)

    return [change_x, change_y, change_z]


def generate_table_wild(study_path, p_num, state):
    print('\n\nReading ActiGraph, watch, and timesheet data...')
    path_acti = study_path + "/" + p_num + "/" + state + "/Actigraph/Clean/" + p_num + ' ' + state + " Freedson.csv"
    path_acti_vm3 = study_path + "/" + p_num + "/" + state + "/Actigraph/Clean/" + p_num + ' ' + state + " VM3.csv"
    path_watch = study_path + "/" + p_num + "/" + state + '/Wrist/Aggregated/Accelerometer/Accelerometer.csv'
    df_acti = pd.read_csv(path_acti, index_col=None, header=1)
    df_acti_vm3 = pd.read_csv(path_acti_vm3, index_col=None, header=1)
    df_watch = pd.read_csv(path_watch, index_col=None, header=0)
    print('Done')

    print('Checking if timesheet exists and reading if it does...')
    path_ts = study_path + "/" + p_num + "/" + state + "/" + p_num + " " + state + " Log.csv"

    if not os.path.exists(path_ts):
        print('Timesheet not found.')
    df_ts = pd.read_csv(path_ts, index_col=None, header=0)
    print('Done')

    # print('Resampling watch data...')
    # df_watch_r = resample(df_watch, 'Time', 20, 100)
    # print('Done')

    print('Adding datetime info...')
    actigraph_add_datetime(df_acti)
    watch_add_datetime(df_watch)
    print('Done')

    print('Writing every minute data into the table...')
    l_state = []
    l_participant = []
    l_intensity = []
    l_mets_freedson = []
    l_mets_vm3 = []
    l_mets_freedson_output = []
    l_mets_vm3_output = []
    l_datetime = []
    l_watch_x_sqd = []
    l_watch_y_sqd = []
    l_watch_z_sqd = []
    l_watch_x_raw = []
    l_watch_y_raw = []
    l_watch_z_raw = []
    x_change_count = []
    y_change_count = []
    z_change_count = []

    for d in range(len(df_ts['Day'])):
        st = pd.to_datetime(df_ts['Start Time'][d], format='%H:%M:%S')
        et = pd.to_datetime(df_ts['End Time'][d], format='%H:%M:%S')
        start_time = pd.Timestamp.combine(pd.to_datetime(df_ts['Start Date'][d], format='%m/%d/%y').date(), st.time())
        for i in range(int((et - st).seconds / 60)):
            end_time = start_time + pd.DateOffset(minutes=1)
            temp = df_watch.loc[(df_watch['Datetime'] >= start_time) & (df_watch['Datetime'] < end_time)].reset_index(
                drop=True)
            temp2 = df_acti.loc[(df_acti['Datetime'] >= start_time) & (df_acti['Datetime'] < end_time)].reset_index(
                drop=True)

            if len(temp['Datetime']) > 0:
                if len(temp2['Datetime']) > 0:
                    l_state.append(state)
                    l_participant.append(p_num)
                    l_intensity.append(get_intensity(df_watch, start_time))
                    l_mets_freedson.append(get_met_freedson(df_acti, start_time))
                    l_mets_vm3.append(get_met_vm3(df_acti, start_time))
                    l_mets_freedson_output.append(get_met_output(df_acti, start_time))
                    l_mets_vm3_output.append(get_met_output(df_acti_vm3, start_time))
                    l_datetime.append(start_time)
                    watch_3d_sqd = get_3d_sqd(df_watch, start_time)
                    watch_3d_raw = get_3d_raw(df_watch, start_time)
                    l_watch_x_sqd.append(watch_3d_sqd[0])
                    l_watch_y_sqd.append(watch_3d_sqd[1])
                    l_watch_z_sqd.append(watch_3d_sqd[2])
                    l_watch_x_raw.append(watch_3d_raw[0])
                    l_watch_y_raw.append(watch_3d_raw[1])
                    l_watch_z_raw.append(watch_3d_raw[2])
                    sign_change_3d = get_sign_change_count(df_watch, start_time)
                    x_change_count.append(sign_change_3d[0])
                    y_change_count.append(sign_change_3d[1])
                    z_change_count.append(sign_change_3d[2])
            start_time += pd.DateOffset(minutes=1)

    the_table = {'Participant': l_participant, 'State': l_state, 'Datetime': l_datetime, 'sum_x_sq_diff': l_watch_x_sqd,
                 'sum_y_sq_diff': l_watch_y_sqd, 'sum_z_sq_diff': l_watch_z_sqd, 'x_mean': l_watch_x_raw,
                 'y_mean': l_watch_y_raw, 'z_mean': l_watch_z_raw, 'x_change_count': x_change_count,
                 'y_change_count': y_change_count, 'z_change_count': z_change_count, 'Watch Intensity': l_intensity,
                 'MET (Freedson)': l_mets_freedson, 'MET (VM3)': l_mets_vm3, 
                 'MET (Freedson output)': l_mets_freedson_output, 'MET (VM3 output)': l_mets_vm3_output}
    print('Done')

    print('Saving the table...')
    df_the_table = pd.DataFrame(the_table)
    df_the_table.to_csv(f"{study_path}/{p_num}/{state}/Summary/Actigraph/{p_num} {state} IntensityMETMinLevel.csv",
                        index=False, encoding='utf8')
    print('Done')


def generate_table_lab(study_path, p_num, state):
    print('\n\nReading ActiGraph, watch, and timesheet data...')
    path_acti = study_path + "/" + p_num + "/" + state + "/Actigraph/Clean/" + p_num + ' ' + state + " Freedson.csv"
    path_acti_vm3 = study_path + "/" + p_num + "/" + state + "/Actigraph/Clean/" + p_num + ' ' + state + " VM3.csv"
    path_watch = study_path + "/" + p_num + "/" + state + '/Wrist/Aggregated/Accelerometer/Accelerometer.csv'
    df_acti = pd.read_csv(path_acti, index_col=None, header=1)
    df_acti_vm3 = pd.read_csv(path_acti_vm3, index_col=None, header=1)
    df_watch = pd.read_csv(path_watch, index_col=None, header=0)
    print('Done')

    print('Checking if timesheet exists and reading if it does...')
    path_ts = study_path + "/" + p_num + "/" + state + "/" + p_num + " " + state + " Log.csv"
    if not os.path.exists(path_ts):
        print('Timesheet not found.')

    df_ts = pd.read_csv(path_ts, index_col=None, header=0)
    print('Done')

    # print('Resampling watch data...')
    # df_watch_r = resample(df_watch, 'Time', 20, 100)
    # print('Done')

    print('Adding datetime info...')
    actigraph_add_datetime(df_acti)
    watch_add_datetime(df_watch)
    print('Done')

    print('Generating The Table...')
    d_mets = {}
    d_mets['breathing']=1.3
    d_mets['computer']=1.3
    d_mets['reading']=1.3
    d_mets['lie down']=1.3
    d_mets['standing']=1.8
    d_mets['sweeping']=2.3
    d_mets['slow walk']=2.8
    d_mets['pushups']=3.8
    d_mets['fast walk']=4.3
    d_mets['squats']=5
    d_mets['running']=6
    d_mets['aerobics']=7.3
    d_mets['stairs']=8

    l_activity = []
    l_minute = []
    l_state = []
    l_participant = []
    l_mets_ainsworth = []
    l_intensity = []
    l_fit = []
    l_mets_freedson = []
    l_mets_vm3 = []
    l_mets_freedson_output = []
    l_mets_vm3_output = []
    l_datetime = []
    l_watch_x_sqd = []
    l_watch_y_sqd = []
    l_watch_z_sqd = []
    l_watch_x_raw = []
    l_watch_y_raw = []
    l_watch_z_raw = []
    x_change_count = []
    y_change_count = []
    z_change_count = []

    for i in range(len(df_ts['Activity'])):
        if not pd.isnull(df_ts['Start Time'][i]):
            st = pd.to_datetime(df_ts['Start Time'][i], format='%H:%M:%S')
            et = pd.to_datetime(df_ts['End Time'][i], format='%H:%M:%S')
            start_time = pd.Timestamp.combine(pd.to_datetime(df_ts['Start Date'][i], format='%m/%d/%y').date(),
                                              st.time())
            for j in range(int((et - st).seconds / 60)):
                l_activity.append(df_ts['Activity'][i])
                l_minute.append(j + 1)
                l_state.append(df_ts['State'][i])
                l_participant.append(p_num)
                l_mets_ainsworth.append(d_mets[df_ts['Activity'][i]])
                l_fit.append((df_ts['End Calorie'][i] - df_ts['Start Calorie'][i]) / 5)
                l_intensity.append(get_intensity(df_watch, start_time))
                l_mets_freedson.append(get_met_freedson(df_acti, start_time))
                l_mets_vm3.append(get_met_vm3(df_acti, start_time))
                l_mets_freedson_output.append(get_met_output(df_acti, start_time))
                l_mets_vm3_output.append(get_met_output(df_acti_vm3, start_time))
                l_datetime.append(start_time)
                watch_3d_sqd = get_3d_sqd(df_watch, start_time)
                watch_3d_raw = get_3d_raw(df_watch, start_time)
                l_watch_x_sqd.append(watch_3d_sqd[0])
                l_watch_y_sqd.append(watch_3d_sqd[1])
                l_watch_z_sqd.append(watch_3d_sqd[2])
                l_watch_x_raw.append(watch_3d_raw[0])
                l_watch_y_raw.append(watch_3d_raw[1])
                l_watch_z_raw.append(watch_3d_raw[2])
                sign_change_3d = get_sign_change_count(df_watch, start_time)
                x_change_count.append(sign_change_3d[0])
                y_change_count.append(sign_change_3d[1])
                z_change_count.append(sign_change_3d[2])
                start_time += pd.DateOffset(minutes=1)

    the_table = {'Participant': l_participant, 'State': l_state, 'Activity': l_activity, 'Minute': l_minute,
                 'Google Fit': l_fit, 'Datetime': l_datetime, 'sum_x_sq_diff': l_watch_x_sqd,
                 'sum_y_sq_diff': l_watch_y_sqd, 'sum_z_sq_diff': l_watch_z_sqd, 'x_mean': l_watch_x_raw,
                 'y_mean': l_watch_y_raw, 'z_mean': l_watch_z_raw, 'x_change_count': x_change_count,
                 'y_change_count': y_change_count, 'z_change_count': z_change_count, 'Watch Intensity': l_intensity,
                 'MET (Ainsworth)': l_mets_ainsworth, 'MET (Freedson)': l_mets_freedson, 'MET (VM3)': l_mets_vm3, 
                 'MET (Freedson output)': l_mets_freedson_output, 'MET (VM3 output)': l_mets_vm3_output}
    df_the_table = pd.DataFrame(the_table)
    df_the_table.to_csv(f"{study_path}/{p_num}/{state}/Summary/Actigraph/{p_num} {state} IntensityMETActivityLevel.csv",
                        index=False, encoding='utf8')
    print('Done')

    print('Writing every minute data into the table...')
    l_state = []
    l_participant = []
    l_intensity = []
    l_mets_freedson = []
    l_mets_vm3 = []
    l_mets_freedson_output = []
    l_mets_vm3_output = []
    l_datetime = []
    l_watch_x_sqd = []
    l_watch_y_sqd = []
    l_watch_z_sqd = []
    l_watch_x_raw = []
    l_watch_y_raw = []
    l_watch_z_raw = []

    start_time = pd.to_datetime(df_ts['Trial Start'][0], format='%m/%d/%y %H:%M:%S')
    # end_time = pd.to_datetime(df_ts['Trial End'][0], format='%m/%d/%y %H:%M:%S')

    st = pd.to_datetime(df_ts['Trial Start'][0], format='%m/%d/%y %H:%M:%S')
    et = pd.to_datetime(df_ts['Trial End'][0], format='%m/%d/%y %H:%M:%S')

    for i in range(int((et - st).seconds / 60)):
        l_state.append(state)
        l_participant.append(p_num)
        l_intensity.append(get_intensity(df_watch, start_time))
        l_mets_freedson.append(get_met_freedson(df_acti, start_time))
        l_mets_vm3.append(get_met_vm3(df_acti, start_time))
        l_mets_freedson_output.append(get_met_output(df_acti, start_time))
        l_mets_vm3_output.append(get_met_output(df_acti_vm3, start_time))
        l_datetime.append(start_time)
        watch_3d_sqd = get_3d_sqd(df_watch, start_time)
        watch_3d_raw = get_3d_raw(df_watch, start_time)
        l_watch_x_sqd.append(watch_3d_sqd[0])
        l_watch_y_sqd.append(watch_3d_sqd[1])
        l_watch_z_sqd.append(watch_3d_sqd[2])
        l_watch_x_raw.append(watch_3d_raw[0])
        l_watch_y_raw.append(watch_3d_raw[1])
        l_watch_z_raw.append(watch_3d_raw[2])
        start_time += pd.DateOffset(minutes=1)

    the_table = {'Participant': l_participant, 'State': l_state, 'Datetime': l_datetime, 'sum_x_sq_diff': l_watch_x_sqd,
                 'sum_y_sq_diff': l_watch_y_sqd, 'sum_z_sq_diff': l_watch_z_sqd, 'x_mean': l_watch_x_raw,
                 'y_mean': l_watch_y_raw, 'z_mean': l_watch_z_raw, 'Watch Intensity': l_intensity,
                 'MET (Freedson)': l_mets_freedson, 'MET (VM3)': l_mets_vm3, 
                 'MET (Freedson output)': l_mets_freedson_output, 'MET (VM3 output)': l_mets_vm3_output}
    print('Done')

    print('Saving the table...')
    df_the_table = pd.DataFrame(the_table)
    df_the_table.to_csv(f"{study_path}/{p_num}/{state}/Summary/Actigraph/{p_num} {state} IntensityMETMinLevel.csv",
                        index=False, encoding='utf8')
    print('Done')


def main():
    # path of study folder
    study_path = str(sys.argv[1])
    # participant# (eg. "P301 P302 P401")
    p_nums = str(sys.argv[2])
    # in-lab or in-wild (eg. "In Lab" or "In Wild")
    state = str(sys.argv[3])

    participants = p_nums.split(' ')

    for p_num in participants:
        print('Running ActiGraph pipeline for '+p_num+' '+state)
        
        # check if weartime file generated by actigraph software has been split into 2 separate files
        check_split_weartime_file(study_path, p_num, state)

        # generate_summary(study_path,p_num,state)
        if state == 'In Lab':
            generate_table_lab(study_path, p_num, state)
        elif state == 'In Wild':
            generate_table_wild(study_path, p_num, state)


if __name__ == '__main__':
    main()
