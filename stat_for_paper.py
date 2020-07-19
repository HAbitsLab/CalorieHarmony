import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
import sys
from scipy.stats import spearmanr


def print_r2_vs_ainsworth(output_path):
    """
    print r2 (vs Ainsworth) for WRIST, ActiGraph VM3, and Google Fit.
    """
    numbers = []
    subfolders = [f.path for f in os.scandir(output_path) if f.is_dir()]
    for f in subfolders:
        if '_out' in f:
            file_path = os.path.join(output_path, f, 'r2_estimation.txt')
            file = open(file_path, 'r')
            num = file.read().split('\n')
            numbers.append(float(num[0]))
            file.close()
    print('WRIST')
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
    print('ActiGraph VM3')
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
    print('Google Fit')
    print('R2 mean: %g' % np.mean(numbers))
    print('R2 var: %g' % np.var(numbers))


def print_r2_wrist_vs_acti(output_path):
    """
    print r2 score for WRIST vs ActiGraph VM3, in-lab and in-wild.
    """
    r2 = []
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith("lab_est_vs_vm3_r2.txt"):
                text_file = open(os.path.join(root, file), 'r')
                r2.append(float(text_file.read().split('\n')[0]))
                text_file.close()
    print('WRIST vs ActiGraph (in-lab) r2:')
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
    print('WRIST vs ActiGraph (in-wild) r2:')
    print('mean:')
    print(np.mean(r2))
    print('std:')
    print(np.std(r2))


def print_stats(output_path):
    """
    print confusion matrix, cohen's kappa, and RMSE values
    """
    tables = []
    subfolders = [f.path for f in os.scandir(output_path) if f.is_dir()]
    for f in subfolders:
        if '_out' in f:
            path_table = os.path.join(output_path, f, 'table_with_estimation.csv')
            df_table = pd.read_csv(path_table, index_col=None, header=0)
            tables.append(df_table)
    df_table_all = pd.concat(tables, sort=False).reset_index(drop=True)
    df_table_all = df_table_all.loc[df_table_all['model_classification'] != -1]

    df_table_all['y_true'] = ''
    df_table_all.loc[df_table_all['MET (Ainsworth)'] <= 1.5, 'y_true'] = 'Sedentary'
    df_table_all.loc[
        (df_table_all['MET (Ainsworth)'] > 1.5) & (df_table_all['MET (Ainsworth)'] < 3), 'y_true'] = 'Light'
    df_table_all.loc[df_table_all['MET (Ainsworth)'] >= 3, 'y_true'] = 'Moderate/Vigorous'

    df_table_all['y_pred_WristML'] = ''
    df_table_all.loc[df_table_all['estimation'] <= 1.5, 'y_pred_WristML'] = 'Sedentary'
    df_table_all.loc[(df_table_all['estimation'] > 1.5) & (df_table_all['estimation'] < 3), 'y_pred_WristML'] = 'Light'
    df_table_all.loc[df_table_all['estimation'] >= 3, 'y_pred_WristML'] = 'Moderate/Vigorous'

    df_table_all['y_pred_VM3'] = ''
    df_table_all.loc[df_table_all['MET (VM3)'] <= 1.5, 'y_pred_VM3'] = 'Sedentary'
    df_table_all.loc[(df_table_all['MET (VM3)'] > 1.5) & (df_table_all['MET (VM3)'] < 3), 'y_pred_VM3'] = 'Light'
    df_table_all.loc[df_table_all['MET (VM3)'] >= 3, 'y_pred_VM3'] = 'Moderate/Vigorous'

    df_table_all['y_pred_GoogleFit'] = ''
    df_table_all.loc[df_table_all['MET (Google Fit)'] <= 1.5, 'y_pred_GoogleFit'] = 'Sedentary'
    df_table_all.loc[
        (df_table_all['MET (Google Fit)'] > 1.5) & (df_table_all['MET (Google Fit)'] < 3), 'y_pred_GoogleFit'] = 'Light'
    df_table_all.loc[df_table_all['MET (Google Fit)'] >= 3, 'y_pred_GoogleFit'] = 'Moderate/Vigorous'

    print('Ainsworth vs Google Fit')
    print('Confusion Matrix:')
    results = confusion_matrix(df_table_all['y_true'], df_table_all['y_pred_GoogleFit'])
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
    kappa = cohen_kappa_score(df_table_all['y_true'], df_table_all['y_pred_GoogleFit'])
    print(kappa)
    print('RMSE value:')
    rmse = mean_squared_error(df_table_all['MET (Ainsworth)'], df_table_all['MET (Google Fit)'], squared=False)
    print(rmse)

    print('Ainsworth vs WristML')
    print('Confusion Matrix:')
    results = confusion_matrix(df_table_all['y_true'], df_table_all['y_pred_WristML'])
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
    kappa = cohen_kappa_score(df_table_all['y_true'], df_table_all['y_pred_WristML'])
    print(kappa)
    print('RMSE value:')
    rmse = mean_squared_error(df_table_all['MET (Ainsworth)'], df_table_all['estimation'], squared=False)
    print(rmse)

    print('Ainsworth vs ActiGraph VM3')
    print('Confusion Matrix:')
    results = confusion_matrix(df_table_all['y_true'], df_table_all['y_pred_VM3'])
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
    kappa = cohen_kappa_score(df_table_all['y_true'], df_table_all['y_pred_VM3'])
    print(kappa)
    print('RMSE value:')
    rmse = mean_squared_error(df_table_all['MET (Ainsworth)'], df_table_all['MET (VM3)'], squared=False)
    print(rmse)

    print('ActiGraph VM3 vs WristML')
    print('Confusion Matrix:')
    results = confusion_matrix(df_table_all['y_pred_VM3'], df_table_all['y_pred_WristML'])
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
    kappa = cohen_kappa_score(df_table_all['y_pred_VM3'], df_table_all['y_pred_WristML'])
    print(kappa)
    print('RMSE value:')
    rmse = mean_squared_error(df_table_all['MET (Ainsworth)'], df_table_all['estimation'], squared=False)
    print(rmse)


def print_spearman_avg(output_path):
    """
    print average spearman
    """
    temp = []
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith("lab_ainsworth_vs_vm3_spearman.txt"):
                text_file = open(os.path.join(root, file), 'r')
                temp.append(float(text_file.read().split('\n')[0]))
                text_file.close()
    print('Ainsworth vs ActiGraph (in-lab) Spearman:')
    print('mean:')
    print(np.mean(temp))
    print('std:')
    print(np.std(temp))

    temp = []
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith("lab_ainsworth_vs_est_spearman.txt"):
                text_file = open(os.path.join(root, file), 'r')
                temp.append(float(text_file.read().split('\n')[0]))
                text_file.close()
    print('Ainsworth vs WRIST (in-lab) Spearman:')
    print('mean:')
    print(np.mean(temp))
    print('std:')
    print(np.std(temp))

    temp = []
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith("lab_ainsworth_vs_google_fit_spearman.txt"):
                text_file = open(os.path.join(root, file), 'r')
                temp.append(float(text_file.read().split('\n')[0]))
                text_file.close()
    print('Ainsworth vs Google Fit (in-lab) Spearman:')
    print('mean:')
    print(np.mean(temp))
    print('std:')
    print(np.std(temp))

    temp = []
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith("lab_est_vs_vm3_spearman.txt"):
                text_file = open(os.path.join(root, file), 'r')
                temp.append(float(text_file.read().split('\n')[0]))
                text_file.close()
    print('WRIST vs ActiGraph (in-lab) Spearman:')
    print('mean:')
    print(np.mean(temp))
    print('std:')
    print(np.std(temp))


    temp = []
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith("wild_est_vs_vm3_spearman.txt"):
                text_file = open(os.path.join(root, file), 'r')
                temp.append(float(text_file.read().split('\n')[0]))
                text_file.close()
    print('WRIST vs ActiGraph (in-wild) Spearman:')
    print('mean:')
    print(np.mean(temp))
    print('std:')
    print(np.std(temp))


def print_spearman_overall(output_path):
    """
    print overall spearman
    """
    print('Overall (combining all data points):')
    temp = []
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith("table_with_estimation.csv"):
                path_table = os.path.join(root, file)
                df = pd.read_csv(path_table, index_col=None, header=0)
                temp.append(df)
    df_table_all = pd.concat(temp).reset_index(drop=True)

    l_vm3_all = df_table_all['MET (VM3)'].tolist()
    l_estimation_all = df_table_all['estimation'].tolist()
    l_ainsworth = df_table_all['MET (Ainsworth)'].tolist()
    l_google_fit = df_table_all['MET (Google Fit)'].tolist()

    l_google_fit = [l_google_fit[i] for i in range(len(l_estimation_all)) if not np.isnan(l_estimation_all[i])]
    l_vm3_all = [l_vm3_all[i] for i in range(len(l_estimation_all)) if not np.isnan(l_estimation_all[i])]
    l_estimation_all = [l_estimation_all[i] for i in range(len(l_estimation_all)) if
                        not np.isnan(l_estimation_all[i])]
    l_ainsworth = [l_ainsworth[i] for i in range(len(l_estimation_all)) if not np.isnan(l_estimation_all[i])]

    l_google_fit = [l_google_fit[i] for i in range(len(l_google_fit)) if l_google_fit[i] >= 0]
    l_ainsworth_gf = [l_ainsworth[i] for i in range(len(l_google_fit)) if l_google_fit[i] >= 0]

    corr, _ = spearmanr(np.array(l_estimation_all), np.array(l_vm3_all))
    print('Spearmans correlation (WRIST vs ActiGraph VM3): %g' % corr)

    corr, _ = spearmanr(np.array(l_ainsworth), np.array(l_vm3_all))
    print('Spearmans correlation (Ainsworth vs ActiGraph VM3): %g' % corr)

    corr, _ = spearmanr(np.array(l_ainsworth), np.array(l_estimation_all))
    print('Spearmans correlation (Ainsworth vs WRIST): %g' % corr)

    corr, _ = spearmanr(np.array(l_ainsworth_gf), np.array(l_google_fit))
    print('Spearmans correlation (Ainsworth vs Google Fit): %g' % corr)


if __name__ == '__main__':
    # path of model output folder
    output_path = str(sys.argv[1])

    print_r2_vs_ainsworth(output_path)

    print_stats(output_path)

    print_r2_wrist_vs_acti(output_path)

    print_spearman_avg(output_path)

    print_spearman_overall(output_path)
