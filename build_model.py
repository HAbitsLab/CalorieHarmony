import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import sys
from sklearn import datasets, linear_model, manifold, metrics
from sklearn.linear_model import LinearRegression
from time import time
import joblib


def get_data_target_table(study_path, p_nums):
    """
    The script preprocessing.py needs to be run first to have the tables generated.
    This function goes through each participants' files and generate the training data needed to build the classification model.
    It outputs training data, training labels, and an aggregated table.
    It also builds a linear regression that maps the intensity to the Ainsworth METs and record the _coef.
    
    Parameters:
        Required:
        - study_path -- the path of the study folder
        - p_nums -- participant numbers separated by space (eg. "P301 P302 P401")
    """

    participants = p_nums.split(' ') 

    data_gyro = []
    target = []
    tables = []

    for p in participants:
        path_table = study_path+'/'+p+'/In Lab/Summary/Actigraph/'+p+' In Lab IntensityMETActivityLevel.csv'
        df_table = pd.read_csv(path_table, index_col=None, header=0)
        tables.append(df_table)

        path_ts = study_path+'/'+p+'/In Lab/'+p+' In Lab Log.csv'
        df_ts = pd.read_csv(path_ts, index_col=None, header=0)
        
        path_gyro = study_path+'/'+p+'/In Lab/Wrist/Aggregated/Gyroscope/Gyroscope_resampled.csv'
        df_gyro = pd.read_csv(path_gyro, index_col=None, header=0)
        df_gyro['Datetime'] = pd.to_datetime(df_gyro['Time'], unit='ms', utc=True).dt.tz_convert('America/Chicago').dt.tz_localize(None)
    
        sedentary_activities = ['breathing','computer','standing','reading','lie down']
        
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
                        
                    else:
                        target.append(1)
                        this_min_gyro = [temp_gyro['rotX'],temp_gyro['rotY'],temp_gyro['rotZ']]
                        data_gyro.append(this_min_gyro)
                        
                    start_time += pd.DateOffset(minutes= 1)

    df_table_all = pd.concat(tables).reset_index(drop=True)

    new_data_gyro = [n for n in data_gyro if np.count_nonzero(np.isnan(n[0]))<(n[0].size/2)]
    new_target_gyro = [target[i] for i in range(len(data_gyro)) if np.count_nonzero(np.isnan(data_gyro[i][0]))<(data_gyro[i][0].size/2)]

    np_data_gyro = np.array(new_data_gyro)
    np_target_gyro = np.array(new_target_gyro)

    np_data_gyro_new = []
    for i in range(len(np_data_gyro)):
        data_i = np.array(np_data_gyro[i])
        np_data_gyro_new.append(data_i[0]+data_i[1]+data_i[2])
    np_data_gyro_new = np.array(np_data_gyro_new)

    print("Hours of data: %.4g" % (float(len(np_target_gyro))/float(60)))

    return np_data_gyro_new, np_target_gyro, df_table_all


def save_intensity_coef(df_table_all, study_path):
    """
    This function takes the aggregated table and build a linear regression model.
    The _coef is saved in a txt file for estimate.py to use.
    
    Parameters:
        Required:
        - df_table_all -- the aggregated table
        - study_path -- the path of the study folder
    """

    l_ainsworth = df_table_all['MET (Ainsworth)'].tolist()
    l_intensity = df_table_all['Watch Intensity'].tolist()
    l_ainsworth = [l_ainsworth[i] for i in range(len(l_intensity)) if not np.isnan(l_intensity[i])]
    l_intensity = [l_intensity[i] for i in range(len(l_intensity)) if not np.isnan(l_intensity[i])]

    ainsworth_reshaped = np.array(l_ainsworth).reshape(-1, 1)
    instensity_reshaped = np.array(l_intensity).reshape(-1, 1)

    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(instensity_reshaped, ainsworth_reshaped - 1.3)

    outf = open(study_path+'intensity_coef.txt', 'a')
    outf.write('%.4g\n' % regr.coef_) 
    outf.close()
    print("intensity_coef (regression coef) = %.4g" % regr.coef_)

                    
def build_classification_model(data, target, study_path):
    """
    This function use the data and targets provided to build a classification model.
    The classification helps improving the estimation of the regression model.
    
    Parameters:
        Required:
        - data -- training data
        - target -- training labels
        - study_path -- the path of the study folder
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

    t0 = time()
    model.fit(data, target)
    t1 = time()
    print("Training Time (minutes): %.4g" % (float(t1 - t0)/float(60)))

    joblib.dump(model, study_path+'/xgbc.dat')

    y_pred = model.predict(data)
    print("Train Accuracy: %.4g" % metrics.accuracy_score(target, y_pred)) 


def main():
    # path of study folder
    study_path = str(sys.argv[1])
    # participants# (eg. "P301 P302 P401")
    p_nums = str(sys.argv[2])

    t0 = time()
    data, target, table = get_data_target_table(study_path, p_nums)
    t1 = time()
    print("Preprocessing Time (minutes): %.4g" % (float(t1 - t0)/float(60)))

    save_intensity_coef(table, study_path)

    build_classification_model(data, target, study_path)



if __name__ == '__main__':
    main()