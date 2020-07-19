import sys
from helper_preprocess import generate_table_lab, generate_table_wild


def main():
    """
    Wrist data needs to be in one csv file, sorted and resampled to 20hz
    This script generates the tables needed to build the model.
    """

    # path of study folder
    study_path = str(sys.argv[1])
    # participant# (eg. "P301 P302 P401")
    p_nums = str(sys.argv[2])
    # in-lab or in-wild (eg. "In Lab" or "In Wild")
    state = str(sys.argv[3])

    participants = p_nums.split(' ')

    for p_num in participants:
        if state == 'In Lab':
            generate_table_lab(study_path, p_num)
        elif state == 'In Wild':
            generate_table_wild(study_path, p_num)


if __name__ == '__main__':
    main()
