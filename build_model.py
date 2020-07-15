import sys
from time import time
from helper_build import get_data_target_table, save_intensity_coef, build_classification_model


def build_both_models(study_path, participants):
    """
    This function builds the regression model and records the _coef, then build and save the classification model.
    
    Parameters:
        :param study_path: the path of the study folder (the folder that contains all participants' folders)
        :param participants: list of participant numbers in str (eg. ["P301","P302","P401"])
    """

    t0 = time()

    data, target, table = get_data_target_table(study_path, participants)

    save_intensity_coef(table, study_path)

    build_classification_model(data, target, study_path)

    t1 = time()
    print("Total model build time: %g minutes" % (float(t1 - t0) / float(60)))


def main():
    # path of study folder
    study_path = str(sys.argv[1])
    # participants# (eg. "P301 P302 P401")
    p_nums = str(sys.argv[2])

    participants = p_nums.split(' ')

    build_both_models(study_path, participants)


if __name__ == '__main__':
    main()
