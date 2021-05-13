# differece to v2 is that in v2 branches was more than data in train frame so i though its reason is that we have
# branches  even at places with no sub-feature , assume that we are at a node , when we want to make a sub-feature
# we even include those sub-features that are excluded in that node because of data shortage at a specific node ,
# so in v3 we do not include such sub-features

import pandas as pd
import numpy as np
from entropy_v1 import info_gain
from entropy_v1 import search_class
from entropy_v1 import search_csv
import copy


def labeler(dataframe_random, tree):
    for i in tree:
        temp_frame = pd.DataFrame.copy(dataframe_random)
        if i['label'] is None:
            for j in i:

                if not ((j == 'depth') or (j == 'label') or (j == 'empty') or (j == 'vote') or (j == 'stp_prune')
                        or (j == 'prune_grp')):
                    temptemp_frame = temp_frame[temp_frame[j] == i[j]]  # give us the rows which have the same
                    # sub-feature as we desire
                    if not temptemp_frame.empty:
                        temp_frame = temptemp_frame
                        temp_frame = temp_frame.drop(columns=j)
                        i['empty'] = 0
                    else:
                        i['empty'] = 1

        arr = temp_frame.pop('label').to_numpy()

        if i['empty'] == 1:  # if we saw that no sufficient data is available for specific sub-feature, then vote
            if np.array(np.where(arr == '<=50K')).shape[1] < (len(arr) - np.array(np.where(arr == '<=50K')).shape[1]):
                i['label'] = '>50K'
            else:
                i['label'] = '<=50K'
        else:  # check if all labels are specific
            if np.array(np.where(arr == '<=50K')).shape[1] == len(arr):
                i['label'] = '<=50K'
            elif np.array(np.where(arr == '>50K')).shape[1] == len(arr):
                i['label'] = '>50K'
            else:
                i['label'] = None
    return tree


def find_root(datafrm_random):
    tree = np.array([])
    dicti = search_csv(datafrm_random)  # do the entropy things here
    dicti = search_class(dicti, datafrm_random)
    inf_g = info_gain(dicti, datafrm_random)
    rt_place = np.argmax(inf_g)  # find the maximum index that shows us the root

    for i in dicti[datafrm_random.columns.tolist()[rt_place + 1]]['trn_features']:  # +1 is because the first column
        # is label
        tree = np.append(tree, {'depth': 1, 'label': None, 'empty': None, 'vote': 0, 'stp_prune': 0, 'prune_grp': 0,
                                datafrm_random.columns.tolist()[rt_place + 1]: i})  # making the tree root with
        # this structure : {'depth': 1, 'label': None, 'empty': None, 'vote': 0, 'stp_prune': 0, 'prune_grp': 0,
        #                    feature : sub-feature}
    tree = labeler(datafrm_random, tree)
    return tree


def continue_tree(random_frame, tree):  # after the root is found ( the tree has been initiated ) , continue the tree
    max_loop = len(random_frame.columns.tolist()) - 1

    cont_or_not = True
    loop = 0
    while (cont_or_not):
        loop = loop + 1
        # print(loop)
        num_of_branch = 0
        ind_delete = np.array([], dtype='int')  # indices(branches) to be deleted after a branch is expanded
        for i in tree:

            num_of_branch = num_of_branch + 1
            # print(num_of_branch)
            temp_frame = pd.DataFrame.copy(random_frame)
            if i['label'] is None:
                ind_delete = np.append(ind_delete, np.argwhere(tree == i))
                for j in i:  # for every sub-branch in a branch

                    if not ((j == 'depth') or (j == 'label') or (j == 'empty') or (j == 'vote') or (j == 'stp_prune')
                            or (j == 'prune_grp')):

                        temptemp_frame = temp_frame[temp_frame[j] == i[j]]  # pick rows in which have the
                        # same sub-feature
                        if not temptemp_frame.empty:
                            temp_frame = temptemp_frame
                            temp_frame = temp_frame.drop(columns=j)
                dicti = search_csv(temp_frame)  # do the following to see which feature is good enough to be
                # the next sub-branch
                dicti = search_class(dicti, temp_frame)
                inf_g = info_gain(dicti, temp_frame)
                rt_place = np.argmax(inf_g)

                for b in dicti[temp_frame.columns.tolist()[rt_place + 1]]['trn_features']:

                    temp_i = copy.deepcopy(i)

                    alpha = temp_frame.columns.tolist()[rt_place + 1]
                    lab_temp_frame = pd.DataFrame.copy(temp_frame)
                    temp_i[alpha] = b
                    lab_temptemp_frame = lab_temp_frame[lab_temp_frame[alpha] == b]
                    if not lab_temptemp_frame.empty:  # see if the branch created is empty to get voted for labling
                        lab_temp_frame = lab_temptemp_frame
                        lab_temp_frame = lab_temp_frame.drop(columns=alpha)
                        temp_i['empty'] = 0
                    else:
                        temp_i['empty'] = 1

                    lab_arr = lab_temp_frame['label'].to_numpy()
                    where_lab_arr_up = np.array(np.where(lab_arr == '<=50K')).shape[1]
                    where_lab_arr_dn = np.array(np.where(lab_arr == '>50K')).shape[1]
                    len_lab_arr = len(lab_arr)
                    if temp_i['empty'] == 1:  # voting for empty
                        if where_lab_arr_up < where_lab_arr_dn:
                            temp_i['label'] = '>50K'
                        else:
                            temp_i['label'] = '<=50K'
                    else:
                        if where_lab_arr_up == len_lab_arr:  # ending place of a branch if this happens
                            temp_i['label'] = '<=50K'
                        elif where_lab_arr_dn == len_lab_arr:
                            temp_i['label'] = '>50K'
                        else:
                            temp_i['label'] = None
                            if loop == max_loop - 1:  # the data have multiple labels at the deepest place, then vote
                                temp_i['vote'] = 1

                                if where_lab_arr_up < where_lab_arr_dn:
                                    temp_i['label'] = '>50K'
                                else:
                                    temp_i['label'] = '<=50K'

                    temp_i['depth'] = temp_i['depth'] + 1

                    tree = np.append(tree, temp_i)

        # print((ind_delete))
        tree = np.delete(tree, ind_delete)

        cont_or_not = False
        for i in tree:
            cont_or_not = cont_or_not or (i['label'] is None)  # uf there are still None labels , continue
        # print(cont_or_not)
        # print(tree)
        # print('tree length is :', len(tree))
    np.save('./tree.npy', tree, allow_pickle=True)  # edited

    return tree

#
# train = pd.read_csv("adult.train.10k.discrete.csv")
#
# random_train = train.sample(frac=1)
#
# tree = []
# tree = find_root(random_train, tree)
# print(tree)
# continue_tree(random_train, tree)
