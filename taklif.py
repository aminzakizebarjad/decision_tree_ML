import pandas as pd
import numpy as np
from tree_maker_v5 import continue_tree
from tree_maker_v5 import find_root
from evaluat_v1 import eval


def process(train_percent):
    test_eval = np.array([])
    train_eval = np.array([])
    for i in np.arange(0, 5):
        print('round', i, 'with percentage', train_percent, 'wait until result')
        train = pd.read_csv("adult.train.10k.discrete.csv")
        train = train.sample(frac=train_percent / 100.)
        # random_train = train.sample(frac=1)
        test = pd.read_csv('adult.test.10k.discrete.csv')

        # making tree
        tree = find_root(train)
        # print(tree)
        tree = continue_tree(train, tree)
        lentgh = len(tree)
        train_eval = np.append(train_eval, eval(train, 100, tree))
        test_eval = np.append(test_eval, eval(test, 100, tree))
        print('round ', i, 'with percentage', train_percent)
        print('train_eval is', train_eval[i], 'test_eval', test_eval[i])
        print('length of tree is ', lentgh)
        print('depth of tree is', tree[-1]['depth'])
    print('average train_eval with percentage', train_percent, 'is', np.divide(np.sum(train_eval), 5))
    print('average test_eval percentage', train_percent, 'is', np.divide(np.sum(test_eval), 5))


def total():
    for i in ([25, 35, 45, 55, 65, 75, 100]):
        process(i)


total()
