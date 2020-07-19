import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn import linear_model, metrics
from time import time
import joblib
import os


def extract_features(gyro_data):
    output = []
    for m in gyro_data:
        temp = []
        for n in m:
            for i in range(int(len(n) / 10)):
                temp2 = np.array(n[i * 10:i * 10 + 10])
                temp.append(np.mean(temp2))
                temp.append(np.var(temp2))
        output.append(temp)
    return np.array(output)


def get_data_target_table(study_path, participants):
    # TODO: can set the gyro vs accel as function parameter
    """
    The script preprocessing.py needs to be run first to have the tables generated.
    This function goes through each participants' files
    and generate the training data needed to build the classification model.
    It outputs training data, training labels, and an aggregated table.

    Parameters:
        :param study_path: the path of the study folder (the folder that contains all participants' folders)
        :param participants: list of participant numbers in str (eg. ["P301","P302","P401"])
    """

    data_training = []
    target = []
    tables = []

    for p in participants:
        path_ts = os.path.join(study_path, p, 'In Lab', p + ' In Lab Log.csv')
        df_ts = pd.read_csv(path_ts, index_col=None, header=0)
        path_table = os.path.join(study_path, p, 'In Lab/Summary/Actigraph/', p + ' In Lab IntensityMETActivityLevel.csv')
        df_table = pd.read_csv(path_table, index_col=None, header=0)
        tables.append(df_table)
        path_df = os.path.join(study_path, p, 'In Lab/Wrist/Aggregated/Gyroscope/Gyroscope_resampled.csv')
        # if using accel data to train
        # path_df = study_path + '/' + p + '/In Lab/Wrist/Aggregated/Accelerometer/Accelerometer_resampled.csv'
        df = pd.read_csv(path_df, index_col=None, header=0)
        df['Datetime'] = pd.to_datetime(df['Time'], unit='ms', utc=True).dt.tz_convert(
            'America/Chicago').dt.tz_localize(None)

        sedentary_activities = ['breathing', 'computer', 'reading', 'lie down']

        for i in range(len(df_ts['Activity'])):
            if not pd.isnull(df_ts['Start Time'][i]):
                st = pd.to_datetime(df_ts['Start Time'][i], format='%H:%M:%S')
                et = pd.to_datetime(df_ts['End Time'][i], format='%H:%M:%S')
                start_time = pd.Timestamp.combine(pd.to_datetime(df_ts['Start Date'][i], format='%m/%d/%y').date(),
                                                  st.time())
                for j in range(int((et - st).seconds / 60)):
                    end_time = start_time + pd.DateOffset(minutes=1)
                    temp = df.loc[
                        (df['Datetime'] >= start_time) & (df['Datetime'] < end_time)].reset_index(drop=True)

                    if df_ts['Activity'][i] in sedentary_activities:
                        target.append(0)
                        this_min_data = [temp['rotX'], temp['rotY'], temp['rotZ']]
                        # if using accel data
                        # this_min_data = [temp['accX'], temp['accY'], temp['accZ']]
                        data_training.append(this_min_data)

                    else:
                        target.append(1)
                        this_min_data = [temp['rotX'], temp['rotY'], temp['rotZ']]
                        # if using accel data
                        # this_min_data = [temp['accX'], temp['accY'], temp['accZ']]
                        data_training.append(this_min_data)

                    start_time += pd.DateOffset(minutes=1)

    df_table_all = pd.concat(tables, sort=False).reset_index(drop=True)

    nan_limit = 4
    new_data_training = [n for n in data_training if np.count_nonzero(np.isnan(n[0])) < nan_limit]
    new_target_training = [target[i] for i in range(len(data_training)) if
                           np.count_nonzero(np.isnan(data_training[i][0])) < nan_limit]

    np_target_training = np.array(new_target_training)

    print("Hours of data: %g" % (float(len(np_target_training)) / float(60)))

    return extract_features(new_data_training), np_target_training, df_table_all


def save_intensity_coef(df_table_all):
    """
    This function takes the aggregated table and build a linear regression model.
    The _coef is saved in a txt file for estimate_and_plot.py to use.

    Parameters:
        :param df_table_all: the aggregated table
    """

    l_ainsworth = df_table_all['MET (Ainsworth)'].tolist()
    l_intensity = df_table_all['Watch Intensity'].tolist()
    l_ainsworth = [l_ainsworth[i] for i in range(len(l_intensity)) if not np.isnan(l_intensity[i])]
    l_intensity = [l_intensity[i] for i in range(len(l_intensity)) if not np.isnan(l_intensity[i])]

    ainsworth_reshaped = np.array(l_ainsworth).reshape(-1, 1)
    instensity_reshaped = np.array(l_intensity).reshape(-1, 1)

    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(instensity_reshaped, ainsworth_reshaped - 1.3)

    outf = open('intensity_coef.txt', 'a')
    outf.write('%g\n' % regr.coef_)
    outf.close()
    print("intensity_coef (regression coef) = %g" % regr.coef_)


def build_classification_model(data, target):
    """
    This function use the data and targets provided to build a classification model.
    The classification helps improving the estimation of the regression model.

    Parameters:
        :param data: training data
        :param target: training labels
    """
    # TODO can move to a settings file (test, then delete if not needed)
    model = XGBClassifier(learning_rate=0.01,
                          n_estimators=400,
                          max_depth=10,
                          min_child_weight=1,
                          gamma=0,
                          subsample=1,
                          colsample_btree=1,
                          scale_pos_weight=1,
                          random_state=7,
                          slient=0,
                          nthread=4
                          )

    t0 = time()
    model.fit(data, target)
    t1 = time()
    print("Training Time (minutes): %g" % (float(t1 - t0) / float(60)))
    joblib.dump(model, 'WRIST.dat')

    y_pred = model.predict(data)
    print("Train Accuracy: %g" % metrics.accuracy_score(target, y_pred))

