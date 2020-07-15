import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
import plotly as py
import plotly.graph_objects as go
from helper_build import extract_features
from scipy.stats import pearsonr, spearmanr


def get_data_target_table(study_path, participants, model):
    """
    This function goes through each participants' files
    and generate the testing data needed to test the classification model.
    It outputs testing data, testing labels, and an aggregated table with model prediction.

    Parameters:
        :param study_path: the path of the study folder (the folder that contains all participants' folders)
        :param participants: list of participant numbers in str (eg. ["P301","P302","P401"])
        :param model: the classification model
    """

    data_gyro = []
    target = []
    tables = []

    for p in participants:
        # TODO Hard coded study paths
        path_ts = study_path + '/' + p + '/In Lab/' + p + ' In Lab Log.csv'
        df_ts = pd.read_csv(path_ts, index_col=None, header=0)
        # TODO Hard coded study paths
        path_gyro = study_path + '/' + p + '/In Lab/Wrist/Aggregated/Gyroscope/Gyroscope_resampled.csv'
        df_gyro = pd.read_csv(path_gyro, index_col=None, header=0)
        df_gyro['Datetime'] = pd.to_datetime(df_gyro['Time'], unit='ms', utc=True).dt.tz_convert(
            'America/Chicago').dt.tz_localize(None)

        sedentary_activities = ['breathing', 'computer', 'reading', 'lie down']
        # TODO Hard coded study paths
        path_table = study_path + '/' + p + '/In Lab/Summary/Actigraph/' + p + ' In Lab IntensityMETActivityLevel.csv'
        df_table = pd.read_csv(path_table, index_col=None, header=0)

        prediction = []

        for i in range(len(df_ts['Activity'])):
            if not pd.isnull(df_ts['Start Time'][i]):
                st = pd.to_datetime(df_ts['Start Time'][i], format='%H:%M:%S')
                et = pd.to_datetime(df_ts['End Time'][i], format='%H:%M:%S')
                start_time = pd.Timestamp.combine(pd.to_datetime(df_ts['Start Date'][i], format='%m/%d/%y').date(),
                                                  st.time())
                for j in range(int((et - st).seconds / 60)):
                    end_time = start_time + pd.DateOffset(minutes=1)
                    temp_gyro = df_gyro.loc[
                        (df_gyro['Datetime'] >= start_time) & (df_gyro['Datetime'] < end_time)].reset_index(drop=True)

                    if df_ts['Activity'][i] in sedentary_activities:
                        target.append(0)
                        this_min_gyro = [temp_gyro['rotX'], temp_gyro['rotY'], temp_gyro['rotZ']]
                        data_gyro.append(this_min_gyro)

                    if df_ts['Activity'][i] not in sedentary_activities:
                        target.append(1)
                        this_min_gyro = [temp_gyro['rotX'], temp_gyro['rotY'], temp_gyro['rotZ']]
                        data_gyro.append(this_min_gyro)

                    if len(temp_gyro['rotX']) != 0:
                        this_min_gyro = [temp_gyro['rotX'], temp_gyro['rotY'], temp_gyro['rotZ']]
                        if np.count_nonzero(np.isnan(this_min_gyro[0])) > 4:  # TODO 4 can be const variable
                            prediction.append(-1)
                        else:
                            model_output = model.predict(extract_features([this_min_gyro]))
                            prediction.append(model_output[0])

                    if len(temp_gyro['rotX']) == 0:
                        prediction.append(-1)

                    start_time += pd.DateOffset(minutes=1)

        df_table['model_classification'] = prediction
        tables.append(df_table)

    new_data_gyro = [n for n in data_gyro if np.count_nonzero(np.isnan(n[0])) < 4]  # TODO 4 can be const variable
    new_target_gyro = [target[i] for i in range(len(data_gyro)) if
                       np.count_nonzero(np.isnan(data_gyro[i][0])) < 4] # TODO 4 can be const variable

    # np_data_gyro = np.array(new_data_gyro)
    np_target_gyro = np.array(new_target_gyro)

    print("Hours of data: %g" % (float(len(np_target_gyro)) / float(60)))

    df_table_all = pd.concat(tables).reset_index(drop=True)

    return extract_features(new_data_gyro), np_target_gyro, df_table_all


def set_realistic_met_estimate(table, study_path): # TODO: why study_path param if not used
    """
    This function adds the rescaled intensity values and the estimation to the table.

    Parameters:
        :param table: the table
        :param study_path: the path of the study folder (the folder that contains all participants' folders)
    """

    outf = open('intensity_coef.txt', 'r')
    str_coef = outf.read().split('\n')
    outf.close()
    float_coef = [float(n) for n in str_coef if len(n) != 0]
    intensity_coef = np.mean(float_coef)
    print("intensity_coef (regression coef) = %g" % intensity_coef)
    table['scaled_intensity'] = table['Watch Intensity'] * intensity_coef + 1.3

    estimation = []
    for i in range(len(table['model_classification'])):
        model_classification = table['model_classification'][i]
        scaled_intensity = table['scaled_intensity'][i]

        if model_classification == -1:
            if scaled_intensity < 1:
                estimation.append(1)
            else:
                estimation.append(scaled_intensity)
        elif model_classification == 0:
            if scaled_intensity < 1:
                estimation.append(1)
            elif scaled_intensity > 1.5:
                estimation.append(1.5)
            else:
                estimation.append(scaled_intensity)
        elif model_classification == 1:
            if scaled_intensity < 1.5:
                estimation.append(1.5)
            else:
                estimation.append(scaled_intensity)

    table['estimation'] = estimation


def plot_results(df_table_all, study_path):  # TODO: why study_path param if not used
    """
    This function takes the values from the table to visualize them.
    It compares the model's estimation, ActiGraph's estimation and Google Fit Estimation to the Ainsworth METs.
    The graphs will be saved under the study folder.

    Parameters:
        :param df_table_all: the aggregated table
        :param study_path: the path of the study folder (the folder that contains all participants' folders)
    """

    l_vm3_all = df_table_all['MET (VM3)'].tolist()
    l_estimation_all = df_table_all['estimation'].tolist()
    l_ainsworth = df_table_all['MET (Ainsworth)'].tolist()
    l_google_fit = df_table_all['MET (Google Fit)'].tolist()

    # remove nan
    l_google_fit = [l_google_fit[i] for i in range(len(l_estimation_all)) if not np.isnan(l_estimation_all[i])]
    l_vm3_all = [l_vm3_all[i] for i in range(len(l_estimation_all)) if not np.isnan(l_estimation_all[i])]
    l_estimation_all = [l_estimation_all[i] for i in range(len(l_estimation_all)) if
                        not np.isnan(l_estimation_all[i])]
    l_ainsworth = [l_ainsworth[i] for i in range(len(l_estimation_all)) if not np.isnan(l_estimation_all[i])]

    # remove negative numbers
    l_google_fit = [l_google_fit[i] for i in range(len(l_google_fit)) if l_google_fit[i] >= 0]
    l_ainsworth_gf = [l_ainsworth[i] for i in range(len(l_google_fit)) if l_google_fit[i] >= 0]

    # calculate Pearson's correlation
    corr, _ = pearsonr(np.array(l_estimation_all), np.array(l_vm3_all))
    print('Pearsons correlation (WRIST vs ActiGraph VM3): %g' % corr)
    outf = open('lab_est_vs_vm3_pearson.txt', 'a')
    outf.write('%g\n' % corr)
    outf.close()
    corr, _ = pearsonr(np.array(l_ainsworth), np.array(l_vm3_all))
    print('Pearsons correlation (Ainsworth vs ActiGraph VM3): %g' % corr)
    outf = open('lab_ainsworth_vs_vm3_pearson.txt', 'a')
    outf.write('%g\n' % corr)
    outf.close()
    corr, _ = pearsonr(np.array(l_ainsworth), np.array(l_estimation_all))
    print('Pearsons correlation (Ainsworth vs WRIST): %g' % corr)
    outf = open('lab_ainsworth_vs_est_pearson.txt', 'a')
    outf.write('%g\n' % corr)
    outf.close()
    corr, _ = pearsonr(np.array(l_ainsworth_gf), np.array(l_google_fit))
    print('Pearsons correlation (Ainsworth vs Google Fit): %g' % corr)
    outf = open('lab_ainsworth_vs_google_fit_pearson.txt', 'a')
    outf.write('%g\n' % corr)
    outf.close()

    # calculate Spearman's correlation
    corr, _ = spearmanr(np.array(l_estimation_all), np.array(l_vm3_all))
    print('Spearmans correlation (WRIST vs ActiGraph VM3): %g' % corr)
    outf = open('lab_est_vs_vm3_spearman.txt', 'a')
    outf.write('%g\n' % corr)
    outf.close()
    corr, _ = spearmanr(np.array(l_ainsworth), np.array(l_vm3_all))
    print('Spearmans correlation (Ainsworth vs ActiGraph VM3): %g' % corr)
    outf = open('lab_ainsworth_vs_vm3_spearman.txt', 'a')
    outf.write('%g\n' % corr)
    outf.close()
    corr, _ = spearmanr(np.array(l_ainsworth), np.array(l_estimation_all))
    print('Spearmans correlation (Ainsworth vs WRIST): %g' % corr)
    outf = open('lab_ainsworth_vs_est_spearman.txt', 'a')
    outf.write('%g\n' % corr)
    outf.close()
    corr, _ = spearmanr(np.array(l_ainsworth_gf), np.array(l_google_fit))
    print('Spearmans correlation (Ainsworth vs Google Fit): %g' % corr)
    outf = open('lab_ainsworth_vs_google_fit_spearman.txt', 'a')
    outf.write('%g\n' % corr)
    outf.close()

    vm3_all_reshaped = np.array(l_vm3_all).reshape(-1, 1)
    estimation_all_reshaped = np.array(l_estimation_all).reshape(-1, 1)
    ainsworth_all_reshaped = np.array(l_ainsworth).reshape(-1, 1)
    google_fit_reshaped = np.array(l_google_fit).reshape(-1, 1)
    ainsworth_all_reshaped_gf = np.array(l_ainsworth_gf).reshape(-1, 1)

    act_dict = {}
    activities = ['breathing', 'computer', 'slow walk', 'fast walk', 'standing', 'squats', 'reading', 'aerobics',
                  'sweeping', 'pushups', 'running', 'lie down', 'stairs']
    for a in activities:
        act_dict[a] = [[], []]
    for i in range(len(df_table_all['Activity'])):
        act_dict[df_table_all['Activity'][i]][0].append(df_table_all['estimation'][i])
        act_dict[df_table_all['Activity'][i]][1].append(df_table_all['MET (Ainsworth)'][i])

    regr = linear_model.LinearRegression()
    regr.fit(estimation_all_reshaped, ainsworth_all_reshaped)
    y_pred = regr.predict(estimation_all_reshaped)
    fig = go.Figure()
    for a in act_dict:
        fig.add_trace(go.Scatter(x=act_dict[a][0], y=act_dict[a][1], mode='markers', name=a))

    y_plot = np.reshape(y_pred, y_pred.shape[0])
    fig.add_trace(go.Scatter(x=l_estimation_all, y=y_plot, mode='lines', name='linear regression',
                             line=dict(color='red', width=4)))
    fig.update_layout(title='Linear Regression',
                      xaxis_title='Estimation',
                      yaxis_title='Ainsworth METs')
    py.offline.plot(fig, filename='LR_estimation.html', auto_open=False)
    outf = open('r2_estimation.txt', 'a')
    outf.write('%g\n' % r2_score(ainsworth_all_reshaped, y_pred))
    outf.close()
    print("The r2 score for our estimation is: %g" % (r2_score(ainsworth_all_reshaped, y_pred)))

    act_dict = {}
    activities = ['breathing', 'computer', 'slow walk', 'fast walk', 'standing', 'squats', 'reading', 'aerobics',
                  'sweeping', 'pushups', 'running', 'lie down', 'stairs']
    for a in activities:
        act_dict[a] = [[], []]
    for i in range(len(df_table_all['Activity'])):
        act_dict[df_table_all['Activity'][i]][0].append(df_table_all['MET (VM3)'][i])
        act_dict[df_table_all['Activity'][i]][1].append(df_table_all['MET (Ainsworth)'][i])

    regr = linear_model.LinearRegression()
    regr.fit(vm3_all_reshaped, ainsworth_all_reshaped)
    y_pred = regr.predict(vm3_all_reshaped)
    fig = go.Figure()
    for a in act_dict:
        fig.add_trace(go.Scatter(x=act_dict[a][0], y=act_dict[a][1], mode='markers', name=a))
    y_plot = np.reshape(y_pred, y_pred.shape[0])
    fig.add_trace(
        go.Scatter(x=l_vm3_all, y=y_plot, mode='lines', name='linear regression', line=dict(color='red', width=4)))
    fig.update_layout(title='Linear Regression',
                      xaxis_title='VM3 METs',
                      yaxis_title='Ainsworth METs')
    py.offline.plot(fig, filename='LR_vm3.html', auto_open=False)
    outf = open('r2_vm3.txt', 'a')
    outf.write('%g\n' % r2_score(ainsworth_all_reshaped, y_pred))
    outf.close()
    print("The r2 score for ActiGraph VM3 is: %g" % (r2_score(ainsworth_all_reshaped, y_pred)))

    act_dict = {}
    activities = ['breathing', 'computer', 'slow walk', 'fast walk', 'standing', 'squats', 'reading', 'aerobics',
                  'sweeping', 'pushups', 'running', 'lie down', 'stairs']
    for a in activities:
        act_dict[a] = [[], []]
    for i in range(len(df_table_all['Activity'])):
        act_dict[df_table_all['Activity'][i]][0].append(df_table_all['MET (Google Fit)'][i])
        act_dict[df_table_all['Activity'][i]][1].append(df_table_all['MET (Ainsworth)'][i])

    regr = linear_model.LinearRegression()
    regr.fit(google_fit_reshaped, ainsworth_all_reshaped_gf)
    y_pred = regr.predict(google_fit_reshaped)
    fig = go.Figure()
    for a in act_dict:
        fig.add_trace(go.Scatter(x=act_dict[a][0], y=act_dict[a][1], mode='markers', name=a))
    y_plot = np.reshape(y_pred, y_pred.shape[0])
    fig.add_trace(
        go.Scatter(x=l_google_fit, y=y_plot, mode='lines', name='linear regression', line=dict(color='red', width=4)))
    fig.update_layout(title='Linear Regression',
                      xaxis_title='Google Fit Calorie Reading',
                      yaxis_title='Ainsworth METs')
    py.offline.plot(fig, filename='LR_google_fit.html', auto_open=False)
    outf = open('r2_google_fit.txt', 'a')
    outf.write('%g\n' % r2_score(ainsworth_all_reshaped_gf, y_pred))
    outf.close()
    print("The r2 score for Google Fit is: %g" % (r2_score(ainsworth_all_reshaped_gf, y_pred)))

    act_dict = {}
    activities = ['breathing', 'computer', 'slow walk', 'fast walk', 'standing', 'squats', 'reading', 'aerobics',
                  'sweeping', 'pushups', 'running', 'lie down', 'stairs']
    for a in activities:
        act_dict[a] = [[], []]
    for i in range(len(df_table_all['Activity'])):
        act_dict[df_table_all['Activity'][i]][0].append(df_table_all['estimation'][i])
        act_dict[df_table_all['Activity'][i]][1].append(df_table_all['MET (VM3)'][i])

    regr = linear_model.LinearRegression()
    regr.fit(estimation_all_reshaped, vm3_all_reshaped)
    y_pred = regr.predict(estimation_all_reshaped)
    fig = go.Figure()
    for a in act_dict:
        fig.add_trace(go.Scatter(x=act_dict[a][0], y=act_dict[a][1], mode='markers', name=a))
    y_plot = np.reshape(y_pred, y_pred.shape[0])
    fig.add_trace(
        go.Scatter(x=l_estimation_all, y=y_plot, mode='lines', name='linear regression', line=dict(color='red', width=4)))
    fig.update_layout(title='Linear Regression',
                      xaxis_title='Estimation',
                      yaxis_title='VM3 METs')
    py.offline.plot(fig, filename='LR_est_vs_vm3_lab.html', auto_open=False)
    outf = open('lab_est_vs_vm3_r2.txt', 'a')
    outf.write('%g\n' % r2_score(vm3_all_reshaped, y_pred))
    outf.close()
    print("The r2 score for in lab estimation vs VM3 is: %g" % (r2_score(vm3_all_reshaped, y_pred)))
