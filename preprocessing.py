import numpy as np
import pandas as pd
import os
import sys


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


def generate_table_wild(study_path, p_num, state):
    print('\n\nReading ActiGraph, watch, and timesheet data...')
    path_acti_freedson = study_path + "/" + p_num + "/" + state + "/Actigraph/Clean/" + p_num + ' ' + state + " Freedson.csv"
    path_acti_vm3 = study_path + "/" + p_num + "/" + state + "/Actigraph/Clean/" + p_num + ' ' + state + " VM3.csv"
    path_accel = study_path + "/" + p_num + "/" + state + '/Wrist/Aggregated/Accelerometer/Accelerometer_resampled.csv'

    df_acti_freedson = pd.read_csv(path_acti_freedson, index_col=None, header=1)
    df_acti_vm3 = pd.read_csv(path_acti_vm3, index_col=None, header=1)
    df_accel = pd.read_csv(path_accel, index_col=None, header=0)
    print('Done')

    print('Checking if timesheet exists and reading if it does...')
    path_ts = study_path + "/" + p_num + "/" + state + "/" + p_num + " " + state + " Log.csv"

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
    l_mets_freedson_output = []
    l_mets_vm3_output = []
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
                    l_mets_freedson_output.append(get_met_output(df_acti_freedson, start_time))
                    l_mets_vm3_output.append(get_met_output(df_acti_vm3, start_time))
                    l_datetime.append(start_time)
            start_time += pd.DateOffset(minutes=1)

    the_table = {'Participant': l_participant, 'State': l_state, 'Datetime': l_datetime, 'Watch Intensity': l_intensity,
                 'MET (Freedson)': l_mets_freedson, 'MET (VM3)': l_mets_vm3,
                 'MET (Freedson output)': l_mets_freedson_output, 'MET (VM3 output)': l_mets_vm3_output}
    print('Done')

    print('Saving the table...')
    df_the_table = pd.DataFrame(the_table)
    df_the_table.to_csv(f"{study_path}/{p_num}/{state}/Summary/Actigraph/{p_num} {state} IntensityMETMinLevel.csv",
                        index=False, encoding='utf8')
    print('Done')


def generate_table_lab(study_path, p_num, state):
    print('\n\nParticipant: ' + p_num)
    print('Reading ActiGraph, watch, and timesheet data...')
    path_acti_freedson = study_path + "/" + p_num + "/" + state + "/Actigraph/Clean/" + p_num + ' ' + state + " Freedson.csv"
    path_acti_vm3 = study_path + "/" + p_num + "/" + state + "/Actigraph/Clean/" + p_num + ' ' + state + " VM3.csv"
    path_accel = study_path + "/" + p_num + "/" + state + '/Wrist/Aggregated/Accelerometer/Accelerometer_resampled.csv'

    df_acti_freedson = pd.read_csv(path_acti_freedson, index_col=None, header=1)
    df_acti_vm3 = pd.read_csv(path_acti_vm3, index_col=None, header=1)
    df_accel = pd.read_csv(path_accel, index_col=None, header=0)
    print('Done')

    print('Checking if timesheet exists and reading if it does...')
    path_ts = study_path + "/" + p_num + "/" + state + "/" + p_num + " " + state + " Log.csv"
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
    d_mets = {'breathing': 1.3, 'computer': 1.3, 'reading': 1.3, 'lie down': 1.3, 'standing': 1.8, 'sweeping': 2.3,
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
    l_mets_freedson_output = []
    l_mets_vm3_output = []
    l_datetime = []

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
                l_intensity.append(get_intensity(df_accel, start_time))
                l_mets_freedson.append(get_met_freedson(df_acti_freedson, start_time))
                l_mets_vm3.append(get_met_vm3(df_acti_freedson, start_time))
                l_mets_freedson_output.append(get_met_output(df_acti_freedson, start_time))
                l_mets_vm3_output.append(get_met_output(df_acti_vm3, start_time))
                l_datetime.append(start_time)
                start_time += pd.DateOffset(minutes=1)

    the_table = {'Participant': l_participant, 'State': l_state, 'Activity': l_activity, 'Minute': l_minute,
                 'Google Fit': l_fit, 'Datetime': l_datetime, 'Watch Intensity': l_intensity,
                 'MET (Ainsworth)': l_mets_ainsworth, 'MET (Freedson)': l_mets_freedson, 'MET (VM3)': l_mets_vm3,
                 'MET (Freedson output)': l_mets_freedson_output, 'MET (VM3 output)': l_mets_vm3_output}
    df_the_table = pd.DataFrame(the_table)
    df_the_table.to_csv(f"{study_path}/{p_num}/{state}/Summary/Actigraph/{p_num} {state} IntensityMETActivityLevel.csv",
                        index=False, encoding='utf8')
    print('Done')


def main():
    """
    utility.py needs to be run before it.
    This script generates the tables needed to build the model.
    """

    # path of study folder
    study_path = str(sys.argv[1])
    # participant# (eg. "P301 P302 P401")
    p_nums = str(sys.argv[2])
    # in-lab or in-wild (eg. "In Lab" or "In Wild")
    state = str(sys.argv[3])

    participants = p_nums.split(' ')

    for p_num in participants:
        if state == 'In Lab':
            generate_table_lab(study_path, p_num, state)
        elif state == 'In Wild':
            generate_table_wild(study_path, p_num, state)


if __name__ == '__main__':
    main()
