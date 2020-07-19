from xgboost import XGBClassifier
import sys
from sklearn import metrics
from time import time
import joblib
from helper_estimate import get_data_target_table, set_realistic_met_estimate, plot_results


def test_and_estimate(study_path, participants):
    """
    The build_model.py script needs to be run first to have the model built.
    This function loads the model and test the performance using the input participants' data.
    It estimates the results and plots the comparison graphs.
    
    Parameters:
        :param study_path: the path of the study folder (the folder that contains all participants' folders)
        :param participants: list of participant numbers in str (eg. ["P301","P302","P401"])
    """

    t0 = time()

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

    model = joblib.load('xgbc.dat')

    data, target, table = get_data_target_table(study_path, participants, model)

    y_pred = model.predict(data)
    outf = open('classification_accuracy.txt', 'a')
    outf.write('%g\n' % metrics.accuracy_score(target, y_pred))
    outf.close()
    print("Test Accuracy: %g" % metrics.accuracy_score(target, y_pred))

    set_realistic_met_estimate(table, study_path)

    table.to_csv('table_with_estimation.csv', index=False, encoding='utf8')

    plot_results(table, study_path)

    t1 = time()
    print("Total estimate and test time: %g minutes" % (float(t1 - t0) / float(60)))


def main():
    # path of study folder
    study_path = str(sys.argv[1])
    # participants# (eg. "P301 P302 P401")
    p_nums = str(sys.argv[2])

    participants = p_nums.split(' ')

    test_and_estimate(study_path, participants)


if __name__ == '__main__':
    main()
