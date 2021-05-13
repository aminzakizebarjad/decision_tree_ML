import pandas as pd
import numpy as np
import copy


def eval(test_frame, percent, tree):
    # percent is the percentage of test data to bring into evaluation
    is_eql = 0
    is_neq = 0

    temp_test = pd.DataFrame.copy(test_frame)
    # temp_test = temp_test.sample(frac=(percent / 100.))  # omit this
    # print(temp_test)
    for fr_index in np.arange(0, len(temp_test.index)):
        # print(fr_index)
        # print(temp_test.ndim)
        # print(fr_index)
        for branch in tree:
            brnch_compatbl = True
            for frac_branch in branch:

                if not (frac_branch == 'label' or frac_branch == 'depth' or frac_branch == 'empty'
                        or frac_branch == 'vote' or frac_branch == 'stp_prune' or frac_branch == 'prune_grp'):
                    brnch_compatbl = brnch_compatbl and (temp_test.iloc[fr_index][frac_branch] == branch[frac_branch])
                    if not brnch_compatbl:  # check if the all branch's frac-branch values are equal to test
                        break  # if false means that this branch is incompatible , try another branch

            # after the compatibility was found that means we have successfully ended the for frac-branch loop , it is
            # time to check if label is also equal or not
            if brnch_compatbl:
                # print(branch['label'])
                # print(temp_test.iloc[fr_index]['label'])
                if branch['label'] == temp_test.iloc[fr_index]['label']:
                    is_eql = is_eql + 1
                else:
                    is_neq = is_neq + 1

                break  # break the for branch loop if branch is compatible , because the desired branch is found

    return float(is_eql) / float(is_neq+is_eql)


# test = pd.read_csv('adult.test.10k.discrete.csv')
# tree = np.load('tree.npy', allow_pickle=True)
# # print(tree)
# result = eval(test, 10, tree)
# print('precision is:', result)
