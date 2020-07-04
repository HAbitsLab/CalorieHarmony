import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
import scipy.stats as st


if __name__ == '__main__':
    output_path = '/Users/wilsonwang/Documents/GitHub/CalorieHarmony/output_files'

    numbers = []
    subfolders = [f.path for f in os.scandir(output_path) if f.is_dir()]
    for f in subfolders:
        if '_out' in f:
            file_path = os.path.join(output_path, f, 'r2_estimation.txt')
            file = open(file_path, 'r')
            num = file.read().split('\n')
            numbers.append(float(num[0]))
            file.close()
    print('WristML')
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

    r2 = []
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith("lab_est_vs_vm3_r2.txt"):
                text_file = open(os.path.join(root, file), 'r')
                r2.append(float(text_file.read().split('\n')[0]))
                text_file.close()
    print('WRIST vs ActiGraph (in-lab) r2:')
    print('mean:')
    print(sum(r2) / len(r2))
    print('max:')
    print(max(r2))
    print('min:')
    print(min(r2))

    r2 = []
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith("wild_est_vs_vm3_r2.txt"):
                text_file = open(os.path.join(root, file), 'r')
                r2.append(float(text_file.read().split('\n')[0]))
                text_file.close()
    print('WRIST vs ActiGraph (in-wild) r2:')
    print('mean:')
    print(sum(r2) / len(r2))
    print('max:')
    print(max(r2))
    print('min:')
    print(min(r2))

    temp = []
    for root, dirs, files in os.walk("/Users/wilsonwang/Documents/GitHub/CalorieHarmony/output_files"):
        for file in files:
            if file.endswith("lab_est_vs_vm3_pearson.txt"):
                text_file = open(os.path.join(root, file), 'r')
                temp.append(float(text_file.read().split('\n')[0]))
                text_file.close()
    print('WRIST vs ActiGraph (in-lab) Pearson:')
    print('mean:')
    print(sum(temp) / len(temp))

    temp = []
    for root, dirs, files in os.walk("/Users/wilsonwang/Documents/GitHub/CalorieHarmony/output_files"):
        for file in files:
            if file.endswith("lab_est_vs_vm3_spearman.txt"):
                text_file = open(os.path.join(root, file), 'r')
                temp.append(float(text_file.read().split('\n')[0]))
                text_file.close()
    print('WRIST vs ActiGraph (in-lab) Spearman:')
    print('mean:')
    print(sum(temp) / len(temp))

    temp = []
    for root, dirs, files in os.walk("/Users/wilsonwang/Documents/GitHub/CalorieHarmony/output_files"):
        for file in files:
            if file.endswith("wild_est_vs_vm3_pearson.txt"):
                text_file = open(os.path.join(root, file), 'r')
                temp.append(float(text_file.read().split('\n')[0]))
                text_file.close()
    print('WRIST vs ActiGraph (in-wild) Pearson:')
    print('mean:')
    print(sum(temp) / len(temp))

    temp = []
    for root, dirs, files in os.walk("/Users/wilsonwang/Documents/GitHub/CalorieHarmony/output_files"):
        for file in files:
            if file.endswith("wild_est_vs_vm3_spearman.txt"):
                text_file = open(os.path.join(root, file), 'r')
                temp.append(float(text_file.read().split('\n')[0]))
                text_file.close()
    print('WRIST vs ActiGraph (in-wild) Spearman:')
    print('mean:')
    print(sum(temp) / len(temp))
