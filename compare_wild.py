import numpy as np
import pandas as pd
import os
from xgboost import XGBClassifier
import sys
from sklearn import linear_model
from sklearn.metrics import r2_score
import plotly as py
import plotly.graph_objects as go
from time import time
import joblib
from helper_build import extract_features
from estimate_and_plot import set_realistic_met_estimate
from scipy.stats import pearsonr, spearmanr


def main():
    # path of study folder
    study_path = str(sys.argv[1])
    # participants# (eg. "P301 P302 P401")
    p_nums = str(sys.argv[2])

    t0 = time()

    participants = p_nums.split(' ')

    for p in participants:
        print('Comparing in wild for '+p)
        current_dir = os.getcwd()
        save_folder = os.path.join(os.getcwd(), 'output_files', 'leave_' + p + '_out')
        if os.path.exists(save_folder):
            os.chdir(save_folder)
        else:
            os.chdir(os.path.join(os.getcwd(), 'output_files', 'using_all'))

        # TODO can move to a settings file
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
        model = joblib.load('xgbc.dat')

        # TODO Hard coded study paths
        path_table = study_path + '/' + p + '/In Wild/Summary/Actigraph/' + p + ' In Wild IntensityMETMinLevel.csv'
        df_table = pd.read_csv(path_table, index_col=None, header=0)
        # TODO Hard coded study paths
        path_gyro = study_path + '/' + p + '/In Wild/Wrist/Aggregated/Gyroscope/Gyroscope_resampled.csv'
        df_gyro = pd.read_csv(path_gyro, index_col=None, header=0)
        df_gyro['Datetime'] = pd.to_datetime(df_gyro['Time'], unit='ms', utc=True).dt.tz_convert(
            'America/Chicago').dt.tz_localize(None)

        prediction = []
        for n in df_table['Datetime']:
            start_time = pd.to_datetime(n)
            end_time = start_time + pd.DateOffset(minutes=1)
            temp_gyro = df_gyro.loc[(df_gyro['Datetime'] >= start_time)
                                    & (df_gyro['Datetime'] < end_time)].reset_index(drop=True)
            if len(temp_gyro['rotX']) == 1200: # TODO what is this 1200? min about of sample needed?
                this_min_gyro = [temp_gyro['rotX'], temp_gyro['rotY'], temp_gyro['rotZ']]
                if np.count_nonzero(np.isnan(this_min_gyro[0])) > 4:  # TODO why 4 can be set to const
                    prediction.append(-1)
                else:
                    model_output = model.predict(extract_features([this_min_gyro]))
                    prediction.append(model_output[0])
            else:
                prediction.append(-1)

        df_table['model_classification'] = prediction

        print("Hours of data: %g" % (float(len(df_table)) / float(60)))

        set_realistic_met_estimate(df_table, study_path)
        df_table.to_csv(p+'_in_wild_comparison.csv', index=False, encoding='utf8')

        l_datetime_all = df_table['Datetime'].tolist()
        l_freedson_all = df_table['MET (Freedson)'].tolist()
        l_vm3_all = df_table['MET (VM3)'].tolist()
        l_estimation_all = df_table['estimation'].tolist()
        l_freedson_all = [l_freedson_all[i] for i in range(len(l_estimation_all)) if not np.isnan(l_estimation_all[i])]
        l_vm3_all = [l_vm3_all[i] for i in range(len(l_estimation_all)) if not np.isnan(l_estimation_all[i])]
        l_estimation_all = [l_estimation_all[i] for i in range(len(l_estimation_all)) if
                            not np.isnan(l_estimation_all[i])]
        l_datetime_all = [l_datetime_all[i] for i in range(len(l_estimation_all)) if not np.isnan(l_estimation_all[i])]
        vm3_all_reshaped = np.array(l_vm3_all).reshape(-1, 1)
        estimation_all_reshaped = np.array(l_estimation_all).reshape(-1, 1)
        freedson_all_reshaped = np.array(l_freedson_all).reshape(-1, 1)


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=l_estimation_all, y=l_vm3_all, mode='markers'))
        regr = linear_model.LinearRegression()
        regr.fit(estimation_all_reshaped, vm3_all_reshaped)
        y_pred = regr.predict(estimation_all_reshaped)
        y_plot = np.reshape(y_pred, y_pred.shape[0])
        fig.add_trace(go.Scatter(x=l_estimation_all, y=y_plot, mode='lines', name='linear regression',
                                 line=dict(color='red', width=4)))
        fig.update_layout(title='Linear Regression',
                          xaxis_title='Estimation',
                          yaxis_title='VM3 METs')
        outf = open('wild_est_vs_vm3_r2.txt', 'a')
        outf.write('%g\n' % r2_score(vm3_all_reshaped, y_pred))
        outf.close()
        print("The r2 score for in wild estimation vs VM3 is: %g" % (r2_score(vm3_all_reshaped, y_pred)))

        # calculate Pearson's correlation
        corr, _ = pearsonr(np.array(l_estimation_all), np.array(l_vm3_all))
        print('Pearsons correlation: %g' % corr)
        outf = open('wild_est_vs_vm3_pearson.txt', 'a')
        outf.write('%g\n' % corr)
        outf.close()
        # calculate Spearman's correlation
        corr, _ = spearmanr(np.array(l_estimation_all), np.array(l_vm3_all))
        print('Spearmans correlation: %g' % corr)
        outf = open('wild_est_vs_vm3_spearman.txt', 'a')
        outf.write('%g\n' % corr)
        outf.close()

        py.offline.plot(fig, filename='in_wild_model_to_vm3.html', auto_open=False)


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=l_estimation_all, y=l_freedson_all, mode='markers'))
        regr = linear_model.LinearRegression()
        regr.fit(estimation_all_reshaped, freedson_all_reshaped)
        y_pred = regr.predict(estimation_all_reshaped)
        y_plot = np.reshape(y_pred, y_pred.shape[0])
        fig.add_trace(go.Scatter(x=l_estimation_all, y=y_plot, mode='lines', name='linear regression',
                                 line=dict(color='red', width=4)))
        fig.update_layout(title='Linear Regression',
                          xaxis_title='Estimation',
                          yaxis_title='Freedson METs')
        outf = open('wild_est_vs_freedson_r2.txt', 'a')
        outf.write('%g\n' % r2_score(freedson_all_reshaped, y_pred))
        outf.close()
        print("The r2 score for in wild estimation vs Freedson is: %g" % (r2_score(freedson_all_reshaped, y_pred)))
        py.offline.plot(fig, filename='in_wild_model_to_freedson.html', auto_open=False)


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=l_datetime_all, y=l_estimation_all, mode='markers', name='model estimation'))
        fig.add_trace(go.Scatter(x=l_datetime_all, y=l_vm3_all, mode='markers', name='actigraph vm3'))
        fig.add_trace(go.Scatter(x=l_datetime_all, y=l_freedson_all, mode='markers', name='actigraph freedson'))
        fig.update_layout(title='Model and ActiGraph Estimation',
                          xaxis_title='Datetime',
                          yaxis_title='MET')
        py.offline.plot(fig, filename='in_wild_comparison.html', auto_open=False)

        os.chdir(current_dir)

    t1 = time()
    print("Total in wild comparison time: %g minutes" % (float(t1 - t0) / float(60)))


if __name__ == '__main__':
    main()
