import sys
from time import time
from build_model import build_both_models
from estimate_and_plot import test_and_estimate
import os


def main():
    # path of study folder
    study_path = str(sys.argv[1])
    # participants# (eg. "P301 P302 P401")
    p_nums = str(sys.argv[2])

    t0 = time()

    participants = p_nums.split(' ')

    for p in participants:
        current_dir = os.getcwd()
        print('\n\nLeaving '+p+' out:')
        save_folder = os.path.join(os.getcwd(), 'output_files', 'leave_' + p + '_out')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        os.chdir(save_folder)
        leftout = [p]
        rest = participants.copy()
        rest.remove(p)
        print('Building:')
        build_both_models(study_path, rest)
        print('Testing:')
        test_and_estimate(study_path, leftout)
        os.chdir(current_dir)

    print('\n\nUsing all:')
    save_folder = os.path.join(os.getcwd(), 'output_files', 'using_all')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    os.chdir(save_folder)
    print('Building:')
    build_both_models(study_path, participants)

    t1 = time()
    print("Total CV Time: %.4g minutes" % (float(t1 - t0)/float(60)))


if __name__ == '__main__':
    main()
