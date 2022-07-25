import pandas as pd
import os
import numpy as np
import pytz
from datetime import datetime
import warnings

def actigraph_add_datetime(actigraph_data):
    """
    Add the datetime of the ActiGraph dataframe to a "Datetime" column.

    :param actigraph_data: ActiGraph dataframe
    """

    datetime = []
    for i in range(len(actigraph_data['date'])):
        date = pd.to_datetime(actigraph_data['date'][i], format='%m/%d/%Y').date()
        time = pd.to_datetime(actigraph_data['epoch'][i], format='%I:%M:%S %p').time()
        temp = pd.Timestamp.combine(date, time)
        datetime.append(temp)
    actigraph_data['Datetime'] = datetime
    return(actigraph_data)


def watch_add_datetime(watch_df):
    """
    Add the datetime of the watch dataframe to a "Datetime" column.
    :param watch_df: watch dataframe
    """

    watch_df['Datetime'] = pd.to_datetime(watch_df['Time'], unit='ms', utc=True).dt.tz_convert(
        'America/Chicago').dt.tz_localize(None)
    return(watch_df)


def get_intensity(watch_df, st):
    """
    Calculate and return the minute level intensity value of given watch using the Panasonic equation.
    :param watch_df: the watch dataframe, used to calculate the intensity value
    :param st: the start time, data of the next minute will be used for calculation
    :return: the intensity value of the next minute using the Panasonic equation
    """

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
    """
    Calculate and return the minute level Freedson MET value for the next minute using ActiGraph data.
    link to paper: https://www.ncbi.nlm.nih.gov/pubmed/9588623
    :param df_acti: the ActiGraph dataframe used for calculating the MET value
    :param st: the start time, data of next minute will be used
    :return: the Freedson MET value
    """

    et = st + pd.DateOffset(minutes=1)
    temp = df_acti.loc[(df_acti['Datetime'] >= st) & (df_acti['Datetime'] < et)].reset_index(drop=True)
    if len(temp['axis1']) > 0:
        met = temp['axis1'][0] * 0.000795 + 1.439008
        return met
    else:
        return np.nan


def get_met_vm3(df_acti, st):
    """
    Calculate and return the minute level VM3 MET value for the next minute using ActiGraph data.
    link to paper: https://www.ncbi.nlm.nih.gov/pubmed/21616714
    :param df_acti: the ActiGraph dataframe used for calculating the MET value
    :param st: the start time, data of next minute will be used
    :return: the VM3 MET value
    """

    et = st + pd.DateOffset(minutes=1)
    temp = df_acti.loc[(df_acti['Datetime'] >= st) & (df_acti['Datetime'] < et)].reset_index(drop=True)
    vm3 = (temp['axis1'][0] ** 2 + temp['axis2'][0] ** 2 + temp['axis3'][0] ** 2) ** 0.5
    met = 0.000863 * vm3 + 0.668876
    return met


def get_met_fitbit(cal, weight):
    return cal * 200 / weight / 3.5


def get_met_matcart(df_met, st):
    et = st + pd.DateOffset(minutes=1)
    temp = df_met.loc[(df_met['Datetime'] >= st) & (df_met['Datetime'] < et)].reset_index(drop=True)
    met = round(np.mean(list(temp['MET(MetCart)'])),3)
    return met
    
    
def get_metcart(ROOT_PATH_FSM, met_cart_dic, p):
    metcart_file = open(ROOT_PATH_FSM + 'P' + str(p) + '/Metcart/P' + str(p) + ' (raw)', 'r')
    lines = metcart_file.readlines()
    l_datetime = []
    l_met_value = []
    for line in lines:
        if((len(line.strip())==72) or (len(line.strip())==80)):
            l_datetime.append(met_cart_dic[str(p)][:10] + ' ' + line.strip()[:8])
            l_met_value.append(line.strip()[21:24])
    df_met = pd.DataFrame({'Datetime':l_datetime, 'MET(MetCart)':l_met_value})
    df_met['Datetime'] = pd.to_datetime(df_met['Datetime'], format='%m/%d/%Y %H:%M:%S')
    df_met['MET(MetCart)'] = pd.to_numeric(df_met['MET(MetCart)'], errors = 'coerce').dropna()
    df_met = df_met[pd.to_numeric(df_met['MET(MetCart)'], errors='coerce').notnull()]
    return(df_met)


def get_train_data(df, st, input_type='gyro'):
    et = st + pd.DateOffset(minutes=1)
    temp = df.loc[(df['Datetime'] >= st) & (df['Datetime'] < et)].reset_index(drop=True)
    if(input_type == 'gyro'):
        this_min_data = [temp['rotX'], temp['rotY'], temp['rotZ']]
    elif(input_type == 'acc'):
        this_min_data = [temp['accX'], temp['accY'], temp['accZ']]
    else:
        print('Invalid input data type')
        return
    return(this_min_data)


def extract_features(data):
    output = []
    for m in data:
        temp = []
        for n in m:
            for i in range(int(len(n) / 10)):
                temp2 = np.array(n[i * 10:i * 10 + 10])
                temp.append(np.mean(temp2))
                temp.append(np.var(temp2))
        output.append(temp)
    return np.array(output)