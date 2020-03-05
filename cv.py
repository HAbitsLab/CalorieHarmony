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
from build_model import build_both_models
from estimate import test_and_estimate



def main():
    # path of study folder
    study_path = str(sys.argv[1])
    # participants# (eg. "P301 P302 P401")
    p_nums = str(sys.argv[2])

    t0 = time()

    participants = p_nums.split(' ')

    for p in participants:
        print('\n\nLeaving '+p+' out:')
        leftout = [p]
        rest = participants.copy()
        rest.remove(p)
        print('Building:')
        build_both_models(study_path,rest)
        print('Testing:')
        test_and_estimate(study_path,leftout)

    t1 = time()
    print("Total CV Time: %.4g minutes" % (float(t1 - t0)/float(60)))



if __name__ == '__main__':
    main()