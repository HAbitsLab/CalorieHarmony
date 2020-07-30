import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
import pingouin as pg


def print_r2_vs_ainsworth(output_path):
    """
    print r2 (vs Ainsworth) for WRIST, ActiGraph VM3, and Google Fit.
    """
    print('\nIN-LAB R2 SCORES WHEN COMPARED VS AINSWORTH:')
    numbers = []
    subfolders = [f.path for f in os.scandir(output_path) if f.is_dir()]
    for f in subfolders:
        if '_out' in f:
            file_path = os.path.join(output_path, f, 'r2_estimation.txt')
            file = open(file_path, 'r')
            num = file.read().split('\n')
            numbers.append(float(num[0]))
            file.close()
    print('\nWRIST')
    print('R2 mean: %g' % np.mean(numbers))
    print('R2 var: %g' % np.var(numbers))

    numbers = []
    subfolders = [f.path for f in os.scandir(output_path) if f.is_dir()]
    for f in subfolders:
        if '_out' in f:
            file_path = os.path.join(output_path, f, 'r2_vm3.txt')
            file = open(file_path, 'r')
            num = file.read().split('\n')
            numbers.append(float(num[0]))
            file.close()
    print('\nActiGraph VM3')
    print('R2 mean: %g' % np.mean(numbers))
    print('R2 var: %g' % np.var(numbers))

    numbers = []
    subfolders = [f.path for f in os.scandir(output_path) if f.is_dir()]
    for f in subfolders:
        if '_out' in f:
            file_path = os.path.join(output_path, f, 'r2_google_fit.txt')
            file = open(file_path, 'r')
            num = file.read().split('\n')
            numbers.append(float(num[0]))
            file.close()
    print('\nGoogle Fit')
    print('R2 mean: %g' % np.mean(numbers))
    print('R2 var: %g' % np.var(numbers))


def print_r2_wrist_vs_acti(output_path):
    """
    print r2 score for WRIST vs ActiGraph VM3, in-lab and in-wild.
    """
    print('\n\nR2 SCORE - WRIST VS ACTIGRAPH VM3:')
    r2 = []
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith("lab_est_vs_vm3_r2.txt"):
                text_file = open(os.path.join(root, file), 'r')
                r2.append(float(text_file.read().split('\n')[0]))
                text_file.close()
    print('\nWRIST vs ActiGraph (in-lab) r2:')
    print('mean:')
    print(np.mean(r2))
    print('std:')
    print(np.std(r2))

    r2 = []
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith("wild_est_vs_vm3_r2.txt"):
                text_file = open(os.path.join(root, file), 'r')
                r2.append(float(text_file.read().split('\n')[0]))
                text_file.close()
    print('\nWRIST vs ActiGraph (in-wild) r2:')
    print('mean:')
    print(np.mean(r2))
    print('std:')
    print(np.std(r2))


def print_diff_lab(df_lab):
    """
    print the mean and sd of differences for in-lab.
    """
    print("\nIN-LAB DIFFERENCE:")

    df_lab['Ainsworth-Est'] = df_lab.apply(lambda x: x['MET (Ainsworth)'] - x['estimation'], axis=1)
    df_lab['Ainsworth-VM3'] = df_lab.apply(lambda x: x['MET (Ainsworth)'] - x['MET (VM3)'], axis=1)
    df_lab['VM3-Est'] = df_lab.apply(lambda x: x['MET (VM3)'] - x['estimation'], axis=1)
    df_lab['Ainsworth-GF'] = df_lab.apply(lambda x: x['MET (Ainsworth)'] - x['MET (Google Fit)'], axis=1)

    md = df_lab['Ainsworth-Est'].mean()
    sd = df_lab['Ainsworth-Est'].std()
    print('\nAinsworth MET vs WRIST')
    print('Mean: %g' % md)
    print('Standard deviation: %g' % sd)

    md = df_lab['Ainsworth-VM3'].mean()
    sd = df_lab['Ainsworth-VM3'].std()
    print('\nAinsworth MET vs Actigraph')
    print('Mean: %g' % md)
    print('Standard deviation: %g' % sd)

    md = df_lab['Ainsworth-GF'].mean()
    sd = df_lab['Ainsworth-GF'].std()
    print('\nAinsworth vs Google Fit')
    print('Mean: %g' % md)
    print('Standard deviation: %g' % sd)

    md = df_lab['VM3-Est'].mean()
    sd = df_lab['VM3-Est'].std()
    print('\nActigraph vs WRIST')
    print('Mean: %g' % md)
    print('Standard deviation: %g' % sd)


def print_diff_wild(df_wild):
    """
    print the mean and sd of differences for in-wild.
    """
    print("\n\nIN-WILD DIFFERENCE:")

    df_wild['VM3-Est'] = df_wild.apply(lambda x: x['MET (VM3)'] - x['estimation'], axis=1)

    md = df_wild['VM3-Est'].mean()
    sd = df_wild['VM3-Est'].std()
    print('\nActigraph vs WRIST')
    print('Mean: %g' % md)
    print('Standard deviation: %g' % sd)


def print_stats(df_lab):
    """
    print confusion matrix, cohen's kappa, and RMSE values
    """
    print("\n\nGETTING CONFUSION MATRIX, COHEN'S KAPPA, AND RMSE VALUES:")

    df_lab['y_true'] = ''
    df_lab.loc[df_lab['MET (Ainsworth)'] <= 1.5, 'y_true'] = 'Sedentary'
    df_lab.loc[
        (df_lab['MET (Ainsworth)'] > 1.5) & (df_lab['MET (Ainsworth)'] < 3), 'y_true'] = 'Light'
    df_lab.loc[df_lab['MET (Ainsworth)'] >= 3, 'y_true'] = 'Moderate/Vigorous'

    df_lab['y_pred_WRIST'] = ''
    df_lab.loc[df_lab['estimation'] <= 1.5, 'y_pred_WRIST'] = 'Sedentary'
    df_lab.loc[(df_lab['estimation'] > 1.5) & (df_lab['estimation'] < 3), 'y_pred_WRIST'] = 'Light'
    df_lab.loc[df_lab['estimation'] >= 3, 'y_pred_WRIST'] = 'Moderate/Vigorous'

    df_lab['y_pred_VM3'] = ''
    df_lab.loc[df_lab['MET (VM3)'] <= 1.5, 'y_pred_VM3'] = 'Sedentary'
    df_lab.loc[(df_lab['MET (VM3)'] > 1.5) & (df_lab['MET (VM3)'] < 3), 'y_pred_VM3'] = 'Light'
    df_lab.loc[df_lab['MET (VM3)'] >= 3, 'y_pred_VM3'] = 'Moderate/Vigorous'

    df_lab['y_pred_GoogleFit'] = ''
    df_lab.loc[df_lab['MET (Google Fit)'] <= 1.5, 'y_pred_GoogleFit'] = 'Sedentary'
    df_lab.loc[
        (df_lab['MET (Google Fit)'] > 1.5) & (df_lab['MET (Google Fit)'] < 3), 'y_pred_GoogleFit'] = 'Light'
    df_lab.loc[df_lab['MET (Google Fit)'] >= 3, 'y_pred_GoogleFit'] = 'Moderate/Vigorous'

    print('\nAinsworth vs WRIST')
    print('Confusion Matrix:')
    results = confusion_matrix(df_lab['y_true'], df_lab['y_pred_WRIST'])
    order = [2, 0, 1]
    results = [results[i] for i in order]
    for j in range(len(results)):
        results[j] = [results[j][i] for i in order]
    results = np.array(results)
    print(results)
    print('Confusion Matrix (normalized):')
    s = np.array([sum(n) for n in results])
    results_normalized = np.array([results[i] / s[i] for i in range(len(s))])
    print(results_normalized)
    print("Cohen's kappa:")
    kappa = cohen_kappa_score(df_lab['y_true'], df_lab['y_pred_WRIST'])
    print(kappa)
    print('RMSE value:')
    rmse = mean_squared_error(df_lab['MET (Ainsworth)'], df_lab['estimation'], squared=False)
    print(rmse)

    print('\nAinsworth vs ActiGraph VM3')
    print('Confusion Matrix:')
    results = confusion_matrix(df_lab['y_true'], df_lab['y_pred_VM3'])
    order = [2, 0, 1]
    results = [results[i] for i in order]
    for j in range(len(results)):
        results[j] = [results[j][i] for i in order]
    results = np.array(results)
    print(results)
    print('Confusion Matrix (normalized):')
    s = np.array([sum(n) for n in results])
    results_normalized = np.array([results[i] / s[i] for i in range(len(s))])
    print(results_normalized)
    print("Cohen's kappa:")
    kappa = cohen_kappa_score(df_lab['y_true'], df_lab['y_pred_VM3'])
    print(kappa)
    print('RMSE value:')
    rmse = mean_squared_error(df_lab['MET (Ainsworth)'], df_lab['MET (VM3)'], squared=False)
    print(rmse)

    print('\nAinsworth vs Google Fit')
    print('Confusion Matrix:')
    results = confusion_matrix(df_lab['y_true'], df_lab['y_pred_GoogleFit'])
    order = [2, 0, 1]
    results = [results[i] for i in order]
    for j in range(len(results)):
        results[j] = [results[j][i] for i in order]
    results = np.array(results)
    print(results)
    print('Confusion Matrix (normalized):')
    s = np.array([sum(n) for n in results])
    results_normalized = np.array([results[i] / s[i] for i in range(len(s))])
    print(results_normalized)
    print("Cohen's kappa:")
    kappa = cohen_kappa_score(df_lab['y_true'], df_lab['y_pred_GoogleFit'])
    print(kappa)
    print('RMSE value:')
    rmse = mean_squared_error(df_lab['MET (Ainsworth)'], df_lab['MET (Google Fit)'], squared=False)
    print(rmse)

    print('\nActiGraph VM3 vs WRIST')
    print('Confusion Matrix:')
    results = confusion_matrix(df_lab['y_pred_VM3'], df_lab['y_pred_WRIST'])
    order = [2, 0, 1]
    results = [results[i] for i in order]
    for j in range(len(results)):
        results[j] = [results[j][i] for i in order]
    results = np.array(results)
    print(results)
    print('Confusion Matrix (normalized):')
    s = np.array([sum(n) for n in results])
    results_normalized = np.array([results[i] / s[i] for i in range(len(s))])
    print(results_normalized)
    print("Cohen's kappa:")
    kappa = cohen_kappa_score(df_lab['y_pred_VM3'], df_lab['y_pred_WRIST'])
    print(kappa)
    print('RMSE value:')
    rmse = mean_squared_error(df_lab['MET (VM3)'], df_lab['estimation'], squared=False)
    print(rmse)


def print_rm_corr(df_lab, df_wild):
    """
    print the rm_corr.
    """
    print("\n\nGETTING RM_CORR:")

    print('\nIN-LAB:')

    print('ActiGraph VM3 vs Ainsworth:')
    g = pg.plot_rm_corr(data=df_lab, x='MET (VM3)', y='MET (Ainsworth)', subject='Participant')
    output = pg.rm_corr(data=df_lab, x='MET (VM3)', y='MET (Ainsworth)', subject='Participant')
    print(output)

    print('\nWRIST vs Ainsworth:')
    g = pg.plot_rm_corr(data=df_lab, x='estimation', y='MET (Ainsworth)', subject='Participant')
    output = pg.rm_corr(data=df_lab, x='estimation', y='MET (Ainsworth)', subject='Participant')
    print(output)

    print('\nGoogle Fit vs Ainsworth:')
    g = pg.plot_rm_corr(data=df_lab, x='MET (Google Fit)', y='MET (Ainsworth)', subject='Participant')
    output = pg.rm_corr(data=df_lab, x='MET (Google Fit)', y='MET (Ainsworth)', subject='Participant')
    print(output)

    print('\nWRIST vs ActiGraph VM3:')
    g = pg.plot_rm_corr(data=df_lab, x='estimation', y='MET (VM3)', subject='Participant')
    output = pg.rm_corr(data=df_lab, x='estimation', y='MET (VM3)', subject='Participant')
    print(output)

    print('\nIN-WILD:')

    print('WRIST vs ActiGraph VM3:')
    g = pg.plot_rm_corr(data=df_wild, x='estimation', y='MET (VM3)', subject='Participant')
    output = pg.rm_corr(data=df_wild, x='estimation', y='MET (VM3)', subject='Participant')
    print(output)


def get_df_lab(output_path):
    """
    return a dataframe that has all the in-lab data.
    """
    temp = []
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith("table_with_estimation.csv"):
                path_table = os.path.join(root, file)
                df = pd.read_csv(path_table, index_col=None, header=0)
                temp.append(df)
    df_table_all = pd.concat(temp).reset_index(drop=True)
    df_table_all = df_table_all.loc[~df_table_all['estimation'].isna()].reset_index()
    df_table_all = df_table_all.loc[df_table_all['MET (Google Fit)'] >= 0].reset_index()
    return df_table_all


def get_df_wild(output_path):
    """
    return a dataframe that has all the in-wild data.
    """
    temp = []
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith("_in_wild_comparison.csv"):
                path_table = os.path.join(root, file)
                df = pd.read_csv(path_table, index_col=None, header=0)
                temp.append(df)
    df_table_all = pd.concat(temp).reset_index(drop=True)
    df_table_all = df_table_all.loc[~df_table_all['estimation'].isna()].reset_index()
    return df_table_all


if __name__ == '__main__':
    output_path = os.path.join(os.getcwd(), 'output_files')

    df_lab = get_df_lab(output_path)
    df_wild = get_df_wild(output_path)

    print('\nIN-LAB DATA IN MINUTES: ')
    print(len(df_lab))
    print('\nIN-WILD DATA IN MINUTES: ')
    print(len(df_wild))

    print_diff_lab(df_lab)

    print_diff_wild(df_wild)

    print_r2_vs_ainsworth(output_path)

    print_r2_wrist_vs_acti(output_path)

    print_stats(df_lab)

    print_rm_corr(df_lab, df_wild)
