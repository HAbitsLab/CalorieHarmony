import pandas as pd
import os
import numpy as np
import pytz
from datetime import datetime
from helper_preprocess import actigraph_add_datetime, watch_add_datetime, get_intensity, get_met_fitbit, get_met_freedson, get_met_vm3, get_metcart, get_met_matcart, get_train_data, extract_features
import warnings

def generate_table(PATH_RESAMPLE_ACC, PATH_RESAMPLE_GYRO, ROOT_PATH_FSM, met_cart_dic, participant_weight, activity_estimate, p):
    try:
        df_acc = pd.read_csv(os.path.join('data_phase_2/'+str(p) + PATH_RESAMPLE_ACC, 'acc_resample.csv'))
        df_gyro = pd.read_csv(os.path.join('data_phase_2/'+str(p) + PATH_RESAMPLE_GYRO, 'gyro_resample.csv'))
        df_actigraph = pd.read_csv(ROOT_PATH_FSM + 'P' + str(p) + '/Actigraph/' + 'P' + str(p) + '_inlab_VM3.csv', skiprows=1)
        df_actigraph = actigraph_add_datetime(df_actigraph)
        df_acc = watch_add_datetime(df_acc)
        df_gyro.columns = ['Time','rotX','rotY','rotZ'] # rename
        df_gyro = watch_add_datetime(df_gyro)
        df_logs = pd.read_excel(ROOT_PATH_FSM + 'P' + str(p) + '/Calorie Harmony Phase 2 Activity Log.xlsx',usecols="A:J",skiprows=1,nrows=24)
        df_logs.columns.values[0] = 'Activity' #rename the first column
        df_logs['Start Date'] = met_cart_dic[str(p)][:10]
        df_met = get_metcart(ROOT_PATH_FSM, met_cart_dic, p)
    except:
        print('Participant', p, 'has invalid data')
        return
    
    l_participant = []
    l_datetime = []
    l_activity = []
    l_fit = []
    l_intensity = []
    l_mets_freedson = []
    l_mets_vm3 = []
    l_mets_ainsworth = []
    l_mets_metcart = []
    l_target = []
    data_training = []
    sedentary_activities = ['Rest', 'Typing on a computer while seated', 'Reading a book or magazine while reclining', 'lie down']


    print('Processing: ', str(p))
    weight = participant_weight[str(p)]

    for i in range(len(df_logs['Activity'])): 
        try:
            st = pd.to_datetime(df_logs['Start Time'][i], format='%H:%M:%S')
            et = pd.to_datetime(df_logs['Expected Stop Time'][i], format='%H:%M:%S')
            start_time = pd.Timestamp.combine(pd.to_datetime(df_logs['Start Date'][i]).date(), st.time())
            for j in range(int((et - st).seconds / 60)):
                l_participant.append(p)
                l_datetime.append(start_time)
                l_activity.append(df_logs['Activity'][i])

                cal = (df_logs['Google Fit Calorie Expenditure Reading at Stop'][i] - df_logs['Google Fit Calorie Expenditure Reading at Start'][i]) / int((et - st).seconds / 60)
                l_fit.append(get_met_fitbit(cal, weight))

                l_intensity.append(get_intensity(df_acc, start_time))
                l_mets_freedson.append(get_met_freedson(df_actigraph, start_time))
                l_mets_vm3.append(get_met_vm3(df_actigraph, start_time))
                l_mets_ainsworth.append(activity_estimate[df_logs['Activity'][i]])
                l_mets_metcart.append(get_met_matcart(df_met, start_time))

                # Acc/Gyro Feature Extraction Part for Activity Classification
                if(df_logs['Activity'][i] in sedentary_activities):
                    l_target.append(0)
                else:
                    l_target.append(1)
                data_training.append(get_train_data(df_gyro, start_time, 'gyro'))           
                start_time += pd.DateOffset(minutes=1)
        except:
            #print('Participant', p, 'Activity', df_logs['Activity'][i], start_time, 'encounters a problem')
            continue

    df_result = pd.DataFrame({'Participant': l_participant, 'Datetime': l_datetime, 'Activity': l_activity, 'Intensity (ACC)': l_intensity,
                  'MET (GoogleFit)': l_fit, 'MET (Freedson)': l_mets_freedson, 'MET (VM3)': l_mets_vm3, 'MET (Ainsworth)': l_mets_ainsworth, 'MET (MetCart)': l_mets_metcart})
    
    np_target_training = np.array(l_target)
    np_training = np.array(data_training)
    
    # validation
    # remove rows that contain nans from result table
    idx_null = df_result.loc[pd.isnull(df_result).any(1), :].index.values
    np_target_training = np.delete(np_target_training, idx_null, axis=0)
    np_training =  np.delete(np_training, idx_null, axis=0)
    df_result = df_result.drop(df_result.index[idx_null]).reset_index(drop=True)
    
    # extract features
    data_train = extract_features(np_training)
    y_train = np_target_training
    temp_train = []
    y_temp = []
    idx_null_2 = []
    for i in range(len(data_train)):
        if(len(data_train[i])!=0):
            temp_train.append(data_train[i])
            y_temp.append(y_train[i])
        else:
            idx_null_2.append(i)
    
    data_train = np.array(temp_train)
    y_train = np.array(y_temp)
    df_result_final = df_result.drop(df_result.index[idx_null_2]).reset_index(drop=True)
    
    return(data_train, y_train, df_result_final)