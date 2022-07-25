import pandas as pd
import os
import numpy as np
import pytz
from datetime import datetime
from sklearn import linear_model, metrics
from xgboost import XGBClassifier
import warnings

def get_intensity_coef(df_table_all, gt_type='MetCart'):
    """
    This function takes the aggregated table and build a linear regression model.
    Parameters:
        :param df_table_all: the aggregated table
    """
    if(gt_type == 'MetCart'):
        l_met = df_table_all['MET (MetCart)'].tolist() 
    else:
        l_met = df_table_all['MET (Ainsworth)'].tolist()
    l_intensity = df_table_all['Intensity (ACC)'].tolist()
    l_met_final = [l_met[i] for i in range(len(l_met)) if not np.isnan(l_met[i]) and not np.isnan(l_intensity[i])]
    l_intensity_final = [l_intensity[i] for i in range(len(l_intensity)) if not np.isnan(l_met[i]) and not np.isnan(l_intensity[i])]

    met_reshaped = np.array(l_met_final).reshape(-1, 1)
    instensity_reshaped = np.array(l_intensity_final).reshape(-1, 1)

    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(instensity_reshaped, met_reshaped - 1.3)
    return(regr.coef_[0][0])


def build_classification_model(data, target):
    """
    This function use the data and targets provided to build a classification model.
    The classification helps improving the estimation of the regression model.
    Parameters:
        :param data: training data
        :param target: training labels
    """
    model = XGBClassifier(learning_rate=0.01,
                          n_estimators=400,
                          max_depth=10,
                          min_child_weight=1,
                          gamma=0,
                          subsample=1,
                          scale_pos_weight=1,
                          random_state=7,
                          nthread=4,
                          eval_metric='mlogloss'
                          )

    model.fit(data, target)
    #joblib.dump(model, 'WRIST.dat')

    y_pred = model.predict(data)
    return(model)


def pred_activity(data, model, table):
    table['model_classification'] = model.predict(data)
    return(table)


def set_realistic_met_estimate(table, coef_list):
    """
    This function adds the rescaled intensity values and the estimation to the table.
    Parameters:
        :param table: the table
    """

    intensity_coef = np.mean(coef_list)
    table['scaled_intensity'] = table['Intensity (ACC)'] * intensity_coef + 1.3

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
    return(table)