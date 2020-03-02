import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import sys
from sklearn import datasets, linear_model, manifold, metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import plotly as py
import plotly.graph_objects as go
from time import time
import joblib


def get_data_target_table(study_path, p_nums, model):
    """
    This function goes through each participants' files and generate the testing data needed to test the classification model.
    It outputs testing data, testing labels, and an aggregated table.
    
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
        path_ts = study_path+'/'+p+'/In Lab/'+p+' In Lab Log.csv'
        df_ts = pd.read_csv(path_ts, index_col=None, header=0)
        
        path_gyro = study_path+'/'+p+'/In Lab/Wrist/Aggregated/Gyroscope/Gyroscope_resampled.csv'
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

    new_data_gyro = [n for n in data_gyro if np.count_nonzero(np.isnan(n[0]))<(n[0].size/2)]
    new_target_gyro = [target[i] for i in range(len(data_gyro)) if np.count_nonzero(np.isnan(data_gyro[i][0]))<(data_gyro[i][0].size/2)]

    np_data_gyro = np.array(new_data_gyro)
    np_target_gyro = np.array(new_target_gyro)

    np_data_gyro_new = []
    for i in range(len(np_data_gyro)):
        data_i = np.array(np_data_gyro[i])
        np_data_gyro_new.append(data_i[0]+data_i[1]+data_i[2])
    np_data_gyro_new = np.array(np_data_gyro_new)

    df_table_all = pd.concat(tables).reset_index(drop=True)
        
    return np_data_gyro_new, np_target_gyro, df_table_all


def add_estimation(table, study_path):
    """
    This function adds the rescaled intensity values and the estimation to the table.
    
    Parameters:
        Required:
        - df_table_all -- the aggragated table
        - study_path -- the path of the study folder
    """

    outf = open(study_path+'/intensity_coef.txt', 'r')
    str_coef = outf.read().split('\n')
    outf.close()
    float_coef = [float(n) for n in str_coef if len(n)!=0]
    # intensity_coef = 0.4062
    intensity_coef = np.mean(float_coef)
    print("intensity_coef (regression coef) = %.4g" % intensity_coef)
    table['scaled_intensity'] = table['Watch Intensity']*intensity_coef + 1.3

    estimation = []
    for i in range(len(table['model_classification'])):
        c = table['model_classification'][i]
        s = table['scaled_intensity'][i]
        
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
                
    table['estimation'] = estimation


def plot_results(df_table_all, study_path):
    """
    This function takes the values from the table to visulize them.
    It compares the model's estimation, ActiGraph's estimation and Google Fit Estimation to the Ainsworth METs.
    The graphs will be saved under the study folder.
    
    Parameters:
        Required:
        - df_table_all -- the aggragated table
        - study_path -- the path of the study folder
    """

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


def test_and_estimate(study_path,p_nums):
    """
    The build_model.py script needs to be run first to have the model built.
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

    t0 = time()
    data, target, table = get_data_target_table(study_path,p_nums,model)
    t1 = time()
    print("Preprocessing time (minutes): %.4g" % (float(t1 - t0)/float(60)))

    y_pred = model.predict(data)
    print("Test Accuracy: %.4g" % metrics.accuracy_score(target, y_pred))

    add_estimation(table, study_path)

    plot_results(table, study_path)



def main():
    # path of study folder
    study_path = str(sys.argv[1])
    # participants# (eg. "P301 P302 P401")
    p_nums = str(sys.argv[2])

    test_and_estimate(study_path,p_nums)



if __name__ == '__main__':
    main()