import numpy as np
import pandas as pd
import os


def watch_add_datetime(watch_df):
    """
    Add the datetime of the watch dataframe to a "Datetime" column.

    :param watch_df: watch dataframe
    """

    watch_df['Datetime'] = pd.to_datetime(watch_df['Time'], unit='ms', utc=True).dt.tz_convert(
        'America/Chicago').dt.tz_localize(None)


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


def cal_to_met(cal, weight):
    return cal * 200 / weight / 3.5


def generate_table_wild(study_path, p_num):
    """
    Generate the in-wild table that has the data needed for the model.

    :param study_path: the path of the study folder where data are located
    :param p_num: participant number
    """
    state = 'In Wild'

    print('\n\nReading ActiGraph, watch, and timesheet data...')
    path_acti_freedson = os.path.join(study_path, p_num, 'In Wild/Actigraph/Clean/', p_num + ' In Wild Freedson.csv')
    path_acti_vm3 = os.path.join(study_path, p_num, 'In Wild/Actigraph/Clean/', p_num + ' In Wild VM3.csv')
    path_accel = os.path.join(study_path, p_num, 'In Wild/Wrist/Aggregated/Accelerometer/Accelerometer_resampled.csv')

    df_acti_freedson = pd.read_csv(path_acti_freedson, index_col=None, header=1)
    df_acti_vm3 = pd.read_csv(path_acti_vm3, index_col=None, header=1)
    df_accel = pd.read_csv(path_accel, index_col=None, header=0)
    print('Done')

    print('Checking if timesheet exists and reading if it does...')
    path_ts = os.path.join(study_path, p_num, 'In Wild', p_num + ' In Wild Log.csv')

    if not os.path.exists(path_ts):
        print('Timesheet not found.')
    df_ts = pd.read_csv(path_ts, index_col=None, header=0)
    print('Done')

    print('Adding datetime info...')
    actigraph_add_datetime(df_acti_freedson)
    actigraph_add_datetime(df_acti_vm3)
    watch_add_datetime(df_accel)
    print('Done')

    print('Writing every minute data into the table...')
    l_state = []
    l_participant = []
    l_intensity = []
    l_mets_freedson = []
    l_mets_vm3 = []
    l_datetime = []

    for d in range(len(df_ts['Day'])):
        st = pd.to_datetime(df_ts['Start Time'][d], format='%H:%M:%S')
        et = pd.to_datetime(df_ts['End Time'][d], format='%H:%M:%S')
        start_time = pd.Timestamp.combine(pd.to_datetime(df_ts['Start Date'][d], format='%m/%d/%y').date(), st.time())
        for i in range(int((et - st).seconds / 60)):
            end_time = start_time + pd.DateOffset(minutes=1)
            temp = df_accel.loc[(df_accel['Datetime'] >= start_time)
                                & (df_accel['Datetime'] < end_time)].reset_index(drop=True)
            temp2 = df_acti_freedson.loc[(df_acti_freedson['Datetime'] >= start_time)
                                         & (df_acti_freedson['Datetime'] < end_time)].reset_index(drop=True)

            if not pd.isnull(temp['accX']).all():
                if len(temp2['Datetime']) > 0:
                    l_state.append(state)
                    l_participant.append(p_num)
                    l_intensity.append(get_intensity(df_accel, start_time))
                    l_mets_freedson.append(get_met_freedson(df_acti_freedson, start_time))
                    l_mets_vm3.append(get_met_vm3(df_acti_freedson, start_time))
                    l_datetime.append(start_time)
            start_time += pd.DateOffset(minutes=1)

    the_table = {'Participant': l_participant, 'State': l_state, 'Datetime': l_datetime, 'Watch Intensity': l_intensity,
                 'MET (Freedson)': l_mets_freedson, 'MET (VM3)': l_mets_vm3}
    print('Done')

    print('Saving the table...')
    df_the_table = pd.DataFrame(the_table)
    df_the_table.to_csv(f"{study_path}/{p_num}/{state}/Summary/Actigraph/{p_num} {state} IntensityMETMinLevel.csv",
                        index=False, encoding='utf8')
    print('Done')


def generate_table_lab(study_path, p_num):
    """
    Generate the in-lab table that has the data needed for the model.

    :param study_path: the path of the study folder where data are located
    :param p_num: participant number
    :param state: 'In Lab' or 'In Wild'
    """
    state = 'In Lab'

    print('\n\nParticipant: ' + p_num)
    print('Reading ActiGraph, watch, and timesheet data...')
    path_acti_freedson = os.path.join(study_path, p_num, 'In Lab/Actigraph/Clean/', p_num + ' In Lab Freedson.csv')
    path_acti_vm3 = os.path.join(study_path, p_num, 'In Lab/Actigraph/Clean/', p_num + ' In Lab VM3.csv')
    path_accel = os.path.join(study_path, p_num, 'In Lab/Wrist/Aggregated/Accelerometer/Accelerometer_resampled.csv')

    df_acti_freedson = pd.read_csv(path_acti_freedson, index_col=None, header=1)
    df_acti_vm3 = pd.read_csv(path_acti_vm3, index_col=None, header=1)
    df_accel = pd.read_csv(path_accel, index_col=None, header=0)
    print('Done')

    print('Checking if timesheet exists and reading if it does...')
    path_ts = os.path.join(study_path, p_num, 'In Lab', p_num + ' In Lab Log.csv')
    if not os.path.exists(path_ts):
        print('Timesheet not found.')

    df_ts = pd.read_csv(path_ts, index_col=None, header=0)
    print('Done')

    print('Adding datetime info...')
    actigraph_add_datetime(df_acti_freedson)
    actigraph_add_datetime(df_acti_vm3)
    watch_add_datetime(df_accel)
    print('Done')

    print('Generating The Table...')
    d_mets = {'breathing': 1, 'computer': 1.3, 'reading': 1.3, 'lie down': 1.3, 'standing': 1.8, 'sweeping': 2.3,
              'slow walk': 2.8, 'pushups': 3.8, 'fast walk': 4.3, 'squats': 5, 'running': 6, 'aerobics': 7.3,
              'stairs': 8}

    l_activity = []
    l_minute = []
    l_state = []
    l_participant = []
    l_mets_ainsworth = []
    l_intensity = []
    l_fit = []
    l_mets_freedson = []
    l_mets_vm3 = []
    l_datetime = []

    path_weights = study_path + "/p_weights.csv"
    df_weights = pd.read_csv(path_weights)

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
                cal = (df_ts['End Calorie'][i] - df_ts['Start Calorie'][i]) / 5
                temp = df_weights.loc[df_weights['Participant'] == p_num].reset_index()
                weight = temp['Weight (kg)'][0]
                l_fit.append(cal_to_met(cal, weight))
                l_intensity.append(get_intensity(df_accel, start_time))
                l_mets_freedson.append(get_met_freedson(df_acti_freedson, start_time))
                l_mets_vm3.append(get_met_vm3(df_acti_freedson, start_time))
                l_datetime.append(start_time)
                start_time += pd.DateOffset(minutes=1)

    the_table = {'Participant': l_participant, 'State': l_state, 'Activity': l_activity, 'Minute': l_minute,
                 'MET (Google Fit)': l_fit, 'Datetime': l_datetime, 'Watch Intensity': l_intensity,
                 'MET (Ainsworth)': l_mets_ainsworth, 'MET (Freedson)': l_mets_freedson, 'MET (VM3)': l_mets_vm3}
    df_the_table = pd.DataFrame(the_table)
    df_the_table.to_csv(f"{study_path}/{p_num}/{state}/Summary/Actigraph/{p_num} {state} IntensityMETActivityLevel.csv",
                        index=False, encoding='utf8')
    print('Done')
