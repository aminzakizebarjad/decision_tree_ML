import pandas as pd
import numpy as np


def search_csv(dtframe):
    whole_dict = {}
    for i in list(dtframe):
        # np.append(trn, train[i].to_numpy())
        # print(type(i))
        trn = (dtframe[i].to_numpy())
        trn_features, trn_count = np.unique(trn, return_counts=True)  # , return_index=True)
        thisdict = {'trn_features': trn_features, 'trn_count': trn_count}  # , 'trn_index': trn_index}
        whole_dict[i] = thisdict  # saving features , sub_features and number of sub-features like below
        # whole_dict = {'feature':{'trn_features': trn_features, 'trn_count': trn_count} .... other features like this}
    return whole_dict


def search_class(dicti, dtfram):
    dtfram_tmp = pd.DataFrame.copy(dtfram)  # copy to stop main data get manipulated
    lbl = dtfram_tmp.pop(dtfram_tmp.columns.tolist()[0]).to_numpy()  # pop label column and cast to numpy
    for i in list(dtfram_tmp):  # i is the column feature
        frm = dtfram_tmp.pop(i).to_numpy()  # pop one column from temporary dataframe and cast it to numpy
        numerator = np.array([], dtype='int64')
        for j in dicti[i]['trn_features']:  # get where in feature column, we have specific sub_feature = j
            num = 0
            where_check = np.where(frm == j)
            where_check = np.array(where_check)
            # print(type(where_check))
            for h in where_check:  # we used two for-loop because the where_check matrix has two dimensions
                for y in h:
                    # print('y=', y)
                    if lbl[y] == '<=50K':  # get the exact label for specific sub_feature
                        num = num + 1  # only numbering labels '<=50K' others can be fetched through the whole
                        # number of a feature
            numerator = np.append(numerator, num)  # putting each sub_feature positive label counts in an array
        dicti[i]["num"] = numerator  # add array to the dictionary for feature i
    return dicti



def info_gain(dictio, dtfra):
    ent = 0
    ent_wh = 0
    ent_whole = []
    inf_gane = []
    sm = np.sum(dictio['label']['trn_count'])  # adding both positive and negetive numbers to get total rows in frame,
    # because we may have used some percentage of dataframe for train
    for i in dictio:  # for every feature in dictionary
        # print(dtfra.columns.tolist()[0])
        if i == dtfra.columns.tolist()[0]:  # if we want to calculate total entropy we need our column to be label
            # print(dictio[i]['trn_count'])
            for j in dictio[i]['trn_count']:  # for every sub-feature counts in feature i
                tmp = j / sm  # get the probability of sub-feature
                if tmp != 0:  # log2 may result nan
                    ent_wh = ent_wh + (tmp * np.log2(tmp)) * -1
            # print(ent_wh)
        else:  # if our dataframe column is not label , is a feature
            less_tn_50k = np.subtract(dictio[i]['trn_count'], dictio[i]['num'])
            mr_tn_50k = dictio[i]['num']
            less_pr_50k = np.divide(less_tn_50k, dictio[i]['trn_count'])  # probability of being below 50k for
            # every sub-feature
            # print(less_pr_50k)
            mr_pr_50k = np.divide(mr_tn_50k, dictio[i]['trn_count'])  # probability of being more than 50k for
            # every sub-feature
            # print(mr_pr_50k)
            feature_probabe = np.divide(dictio[i]['trn_count'], sm)  # get the probability of every sub-feature
            res_ls_pr = []
            res_mr_pr = []
            for el_ls in less_pr_50k:
                if el_ls != 0:
                    res_ls_pr = np.append(res_ls_pr, el_ls * np.log2(el_ls))
                else:
                    res_ls_pr = np.append(res_ls_pr, 0)

            for el_mr in mr_pr_50k:
                if el_mr != 0:
                    res_mr_pr = np.append(res_mr_pr, el_mr * np.log2(el_mr))
                else:
                    res_mr_pr = np.append(res_mr_pr, 0)

            each_ent = np.multiply(np.add(res_ls_pr, res_mr_pr), -1)  # get entropy of a sub-feature

            ent = np.sum(np.multiply(feature_probabe, each_ent))  # get entropy of a feature
            ent_whole = np.append(ent_whole, ent)  # appending feature entropy to ent_whole array
    inf_gane = np.subtract(ent_wh, ent_whole)  # inf-gain array of train data
    # print(inf_gane)
    return inf_gane
