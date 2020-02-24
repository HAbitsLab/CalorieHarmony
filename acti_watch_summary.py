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


# due to daylight saving any wrist run after 11/3 need to be rerun to have the right time
# to-do: detect timezone and change time by timezone
def watch_add_datetime(watch_df):
    watch_df['Datetime'] = pd.to_datetime(watch_df['Time'], unit='ms') + pd.DateOffset(hours=-6)

def actigraph_add_datetime(actigraph_data):
    datetime = []
    for i in range(len(actigraph_data['date'])):
        date = pd.to_datetime(actigraph_data['date'][i], format='%m/%d/%Y').date()
        time = pd.to_datetime(actigraph_data['epoch'][i], format='%I:%M:%S %p').time()
        temp = pd.Timestamp.combine(date, time)
        datetime.append(temp)
    actigraph_data['Datetime'] = datetime

def plot_data_watch_without_ts(df_watch, study_path, p_num, state):
    df_watch_20 = df_watch.loc[df_watch.index % 20 == 0].reset_index(drop=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_watch_20['Datetime'], y=df_watch_20['accX'], mode='lines', name='x-axis'))
    fig.add_trace(go.Scatter(x=df_watch_20['Datetime'], y=df_watch_20['accY'], mode='lines', name='y-axis'))
    fig.add_trace(go.Scatter(x=df_watch_20['Datetime'], y=df_watch_20['accZ'], mode='lines', name='z-axis'))

    fig.write_image(study_path+"/"+p_num+"/"+state+"/ActiGraph/"+p_num+' '+state+' watch.png')  
    # fig.show()

def plot_data_acti_without_ts(df_acti, study_path, p_num, state):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_acti['Datetime'], y=df_acti['axis2'], mode='lines', name='x-axis'))
    fig.add_trace(go.Scatter(x=df_acti['Datetime'], y=df_acti['axis1'], mode='lines', name='y-axis'))
    fig.add_trace(go.Scatter(x=df_acti['Datetime'], y=df_acti['axis3'], mode='lines', name='z-axis'))

    fig.write_image(study_path+"/"+p_num+"/"+state+"/ActiGraph/"+p_num+' '+state+' actigraph.png')  
    # fig.show()

def plot_both(df_acti, df_watch, study_path, p_num, state):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_acti['Datetime'], y=df_acti['axis2']/100, mode='lines', name='ActiGraph x-axis'))
    fig.add_trace(go.Scatter(x=df_acti['Datetime'], y=df_acti['axis1']/100, mode='lines', name='ActiGraph y-axis'))
    fig.add_trace(go.Scatter(x=df_acti['Datetime'], y=df_acti['axis3']/100, mode='lines', name='ActiGraph z-axis'))

    df_watch_20 = df_watch.loc[df_watch.index % 20 == 0].reset_index(drop=True)
    fig.add_trace(go.Scatter(x=df_watch_20['Datetime'], y=df_watch_20['accX']-60, mode='lines', name='Watch x-axis'))
    fig.add_trace(go.Scatter(x=df_watch_20['Datetime'], y=df_watch_20['accY']-60, mode='lines', name='Watch y-axis'))
    fig.add_trace(go.Scatter(x=df_watch_20['Datetime'], y=df_watch_20['accZ']-60, mode='lines', name='Watch z-axis'))

    fig.write_image(study_path+"/"+p_num+"/"+state+"/ActiGraph/"+p_num+' '+state+' watch and actigraph.png')  
    fig.show()

def calc_reliability(timeArr, outfolder, unit='second', plot=0):
    """
        Calculate the reliability of time series sensor data in each 'unit' from the start of the 'timeArr' to the end.
        Plot is optional.
        
        Requirement: timeArr must be unixtimestamp in milliseconds.
        
        :param timeArr: time array of unixtimestamp in milliseconds, size N*1
        :param unit: str, options: "second", "minute", "hour"
        :param plot: 0 or 1
        :return countDf: reliability result dataframe with columns 'Time' and 'SampleCounts'.
        
        """
    
    # ==================================================================================
    # generate the reliability dataframe
    # ==================================================================================
    msecCnts = {
        "second": 1000,
        "minute": 60000,
        "hour": 3600000,
    }
    
    timeNoDuplicateArr = np.unique(timeArr)
    timeNoDuplicateArr = np.floor_divide(timeNoDuplicateArr, msecCnts[unit]).astype(int)
    timeNoDuplicateArr = np.sort(timeNoDuplicateArr)

    reliabilityTimeList = []
    reliabilitySampleCountsList = []
    count = 0
    
    # loop through the timeNoDuplicateArr
    for i in range(len(timeNoDuplicateArr) - 1):
        # if next timestamp is the same as current one
        if timeNoDuplicateArr[i+1] == timeNoDuplicateArr[i]:
            count += 1
        else:
            reliabilityTimeList.append(timeNoDuplicateArr[i])
            # count+1 instead of count because it's looking at the next second
            reliabilitySampleCountsList.append(count+1)
            count = 0
            # if the data have a gap, which means unit(s) with no data exist(s)
            for time in range(timeNoDuplicateArr[i] + 1, timeNoDuplicateArr[i+1]):
                reliabilityTimeList.append(time)
                # append 0 to noData seconds
                reliabilitySampleCountsList.append(0)

    # With try-except, no need to check the input and the empty countDf will be returned.
    # Advantage: In batch processing, when empty data files with only header exist,
    #  the reliability files will follow the same pattern.
    try:
        reliabilityTimeList.append(timeNoDuplicateArr[-1])
        reliabilitySampleCountsList.append(count + 1)
    except:
        print('Warning: timeArr is empty!')

    countDf = pd.DataFrame({'Time':reliabilityTimeList,'SampleCounts':reliabilitySampleCountsList},\
                               columns=['Time','SampleCounts'])

    # ==================================================================================
    # plot figure
    # ==================================================================================
    if plot:
        countDf['Time'] = pd.to_datetime(countDf['Time'], unit='s', utc=True) + pd.DateOffset(hours=-5)
        countDf = countDf.set_index(['Time'])
        # countDf.index = countDf.index.tz_convert('US/Central')
        
        f = plt.figure(figsize=(12,5))
        countDf.plot(style=['b-'], ax=f.gca())
        plt.title('Reliability Test')
        plt.ylabel('Count Per Unit')
        # plt.show()
        plt.savefig(os.path.join(outfolder, 'reliability(frequency).png')) # FYI, save fig function
        
    return countDf

def score_reliability(countDf, sensorFreq, unit='second'):
    """
        Calculate and print out the reliability score.
        
        :param countDf: the dataframe of ['Time','SampleCounts'], which is the return of function calc_reliability
        :param sensorFreq: in Hz
        :param unit: str, options: "second", "minute", "hour"
        :return: reliabilityHasDataUnits: the average reliability score for the units that has data
        :return: reliability: the average reliability score for all the units
        """
    
    # ==================================================================================
    # print out reliability score for has-data units and for all units
    # ==================================================================================
    
    countDf.index = pd.to_datetime(countDf.index, unit='s', utc=True) + pd.DateOffset(hours=-5)
    print("Duration: {}".format(countDf.index[-1] - countDf.index[0]))
    
    idealCntInUnits = {
        "second": sensorFreq,
        "minute": sensorFreq*60,
        "hour": sensorFreq*3600,
    }
    countHasDataUnitsDf = countDf[countDf.SampleCounts != 0]
    countHasDataUnitsArr = countHasDataUnitsDf.SampleCounts.values
    samplingFreqArr = np.full_like(countHasDataUnitsArr, idealCntInUnits[unit])
    countHasDataUnitsArr = np.minimum(countHasDataUnitsArr, samplingFreqArr)
    reliabilityHasDataUnits = np.sum(countHasDataUnitsArr)/np.sum(samplingFreqArr)

    print("Reliability for has-data units(seconds or minutes or hours): {}".format(reliabilityHasDataUnits))

    countArr = countDf.SampleCounts.values
    samplingFreqArr = np.full_like(countArr, idealCntInUnits[unit])
    countArr = np.minimum(countArr, samplingFreqArr)
    reliability = np.sum(countArr)/np.sum(samplingFreqArr)
    
    print("Reliability for all the units(seconds or minutes or hours): {}".format(reliability))
    
    return reliabilityHasDataUnits, reliability

def get_watch_weartime(watch_df):
    everymin = []
    sqd = []
    thismin = watch_df['Datetime'][0]
    while thismin < watch_df['Datetime'][len(watch_df['Datetime'])-1]:
        nextmin = thismin + pd.DateOffset(minutes=1)
        temp = watch_df.loc[(watch_df['Datetime'] >= thismin) & (watch_df['Datetime'] < nextmin)].reset_index(drop=True)
        if len(temp['Datetime'])>1:
            parta = temp[:len(watch_df['accX'])].reset_index(drop=True)
            partb = temp[1:].reset_index(drop=True)
            sum_sqd_diff = (partb['accX'] - parta['accX'])**2 + (partb['accY'] - parta['accY'])**2 + (partb['accZ'] - parta['accZ'])**2
            sqd.append(sum_sqd_diff.mean())
            everymin.append(nextmin)
        thismin = nextmin
    
    wearing = {}
    not_wearing = {}
    count_wear = 0
    count_not_wear = 0
    check_wear = False
    check_not_wear = True
    for i in range(len(sqd)):
        if sqd[i] < 0.005:
            lastmin = everymin[i] + pd.DateOffset(minutes=-1)
            count_not_wear += 1
            count_wear = 0
            if count_not_wear >= 60:
                check_not_wear = True
                check_wear = False
            date = lastmin.date().strftime('%m/%d/%Y')
            if check_not_wear:
                if date in not_wearing:
                    not_wearing[date] += 1
                else:
                    not_wearing[date] = 1
            if check_wear:
                if date in wearing:
                    wearing[date] += 1
                else:
                    wearing[date] = 1
        else:
            lastmin = everymin[i] + pd.DateOffset(minutes=-1)
            count_wear += 1
            count_not_wear = 0
            if count_wear >= 3:
                check_wear = True
                check_not_wear = False
            date = lastmin.date().strftime('%m/%d/%Y')
            if check_not_wear:
                if date in not_wearing:
                    not_wearing[date] += 1
                else:
                    not_wearing[date] = 1
            if check_wear:
                if date in wearing:
                    wearing[date] += 1
                else:
                    wearing[date] = 1

    return wearing


def generate_summary(study_path,p_num,state):

    print('\n\nReading ActiGraph, watch, and timesheet data...')
    path_acti = study_path+"/"+p_num+"/"+state+"/ActiGraph/"+p_num+' '+state+" Freedson.csv"
    path_watch = study_path+"/"+p_num+"/"+state+'/Wrist/Aggregated/Accelerometer/Accelerometer.csv'
    df_acti = pd.read_csv(path_acti, index_col=None, header=1)
    df_watch = pd.read_csv(path_watch, index_col=None, header=0)
    print('Done')

    print('Checking if timesheet exists and reading if it does...')
    path_ts = study_path+"/"+p_num+"/"+state+"/"+p_num+" "+state+".csv"
    if not os.path.exists(path_ts):
        print('Timesheet not found.')
    df_ts = pd.read_csv(path_ts, index_col=None, header=0)
    print('Done')

    print('Adding datetime info...')
    actigraph_add_datetime(df_acti)
    watch_add_datetime(df_watch)
    print('Done')


    st = pd.to_datetime(df_ts['Trial Start Time'][0], format='%H:%M:%S')
    if state == 'In Lab':
        et = pd.to_datetime(df_ts['Trial End Time'][0], format='%H:%M:%S')
    elif state == 'In Wild':
        et = pd.to_datetime(df_ts['Trial End Time'][len(df_ts['Date'])-1], format='%H:%M:%S')


    trial_st = pd.Timestamp.combine(pd.to_datetime(df_ts['Date'][0], format='%m/%d/%y').date(), st.time())
    trial_et = pd.Timestamp.combine(pd.to_datetime(df_ts['Date'][len(df_ts['Date'])-1], format='%m/%d/%y').date(), et.time())
    df_acti_trial = df_acti.loc[(df_acti['Datetime'] >= trial_st) & (df_acti['Datetime'] < trial_et)].reset_index(drop=True)
    df_watch_trial = df_watch.loc[(df_watch['Datetime'] >= trial_st) & (df_watch['Datetime'] < trial_et)].reset_index(drop=True)


    summarytxt = open(study_path+"/"+p_num+"/"+state+"/ActiGraph/actigraph_and_wrist_summary.txt", "w")

    print('\n\nActiGraph:')
    summarytxt.write('\n\nActiGraph:') 
    df_acti_weartime = pd.read_csv(study_path+'/'+p_num+'/'+state+'/ActiGraph/'+p_num+' '+state+' Weartime.csv', index_col=None, header=1)
    ind = df_acti_weartime['Date'][df_acti_weartime['Date'] == 'Date/Time Start'].index[0]
    df_acti_weartime = df_acti_weartime.loc[df_acti_weartime.index < ind].reset_index(drop=True)
    for i in range(len(df_acti_weartime['Date'])):
        hours = int(df_acti_weartime['Wear Time (minutes)'][i])/60
        print("On "+df_acti_weartime['Date'][i]+", the wear time is: "+str(hours)+" hours.")
        summarytxt.write("\nOn "+df_acti_weartime['Date'][i]+", the wear time is: "+str(hours)+" hours.")

    print('Plotting ActiGraph data...')
    plot_data_acti_without_ts(df_acti_trial,study_path,p_num,state)
    print('Done')

    print('Plotting Watch data...')
    plot_data_watch_without_ts(df_watch_trial,study_path,p_num,state)
    print('Done')

    print('Plotting both together...')
    plot_both(df_acti_trial,df_watch_trial,study_path,p_num,state)
    print('Done')

    print('\nWrist:')
    summarytxt.write('\n\nWrist:') 
    wearing = get_watch_weartime(df_watch_trial)
    for k in wearing:
        print('On '+k+', there are '+str(wearing[k]/60)+' hours of watch wear time.')
        summarytxt.write('\nOn '+k+', there are '+str(wearing[k]/60)+' hours of watch wear time.') 

    print('\nCalculating Watch Data Reliability:')
    summarytxt.write('\n\nWatch Data Reliability:') 
    saveFolder = study_path+"/"+p_num+"/"+state+"/ActiGraph/watch reliability"
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    timeArr = df_watch_trial.iloc[:,0].values
    # requirement: unixtimestamp must be in milliseconds
    countDf = calc_reliability(timeArr, saveFolder, 'second', plot=1)
    countDf.to_csv(os.path.join(saveFolder, 'watch reliability.csv'), index=False)
    scores = score_reliability(countDf, 20, 'second')
    summarytxt.write("\nReliability for has-data units(seconds or minutes or hours): {}".format(scores[0]))
    summarytxt.write("\nReliability for all the units(seconds or minutes or hours): {}".format(scores[1]))


    summarytxt.close() 



def main():

    # path of study folder
    study_path = str(sys.argv[1])
    # participant# (eg. P206)
    p_num = str(sys.argv[2])
    # in-lab or in-wild (eg. In Lab or In Wild)
    state = str(sys.argv[3])

    generate_summary(study_path,p_num,state)




if __name__ == '__main__':
    main()