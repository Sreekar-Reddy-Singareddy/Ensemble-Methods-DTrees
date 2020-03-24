# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 1 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import statistics as st
import pandas as pd
import math
from sklearn import tree as sk_tree
from sklearn.metrics import confusion_matrix
import graphviz

def partition(x):
    columns = list(x)
    del (columns[0])
    dict_all = {}
    for c in columns:
        values = list(x[c].unique())
        new_dict = {};
        for i in values:
            new_dict[c, i] = list(x.loc[x[c] == i].index)
            dict_all = {**dict_all, **new_dict}
    return dict_all

def entropy(y):
    if len(y) == 0:
        return 0
    else:
        p0 = len([c for c in y if c == 0]) / len(y)
        p1 = 1 - p0
        if p0 == 0.0 or p0 == 1.0:
            return 0
        else:
            return -((p0 * math.log(p0, 2)) + (p1 * math.log(p1, 2)))

def mutual_information(x, y):
    e_root = entropy(y['class'])
    y_true = y[y.index.isin(x)]
    y_false = y[~y.index.isin(x)]
    e_true = entropy(y_true['class'])
    e_false = entropy(y_false['class'])
    e_tot = (len(y_true) / len(y)) * e_true + (len(y_false) / len(y)) * e_false
    return e_root - e_tot

def dectree(df, dict, depth):
    if len(dict.keys()) == 0 or len(df['class'].unique()) == 1 or depth == 0:
        if len(df.loc[df['class'] == 1]) >= len(df.loc[df['class'] == 0]):
            return 1
        else:
            return 0
    else:
        mutinfo = {}
        for k in dict.keys():
            node = dict[k]
            mutinfo[k] = mutual_information(node, df)
        splitnode = ()
        Gain = 0
        for k in mutinfo.keys():
            if (mutinfo[k]) >= Gain:
                Gain = mutinfo[k]
                splitnode = k
        feature = splitnode[0]
        value = splitnode[1]
        dict.pop(splitnode)
        depth -= 1
        if len(df.loc[df[feature] == value]) == 0:
            if len(df.loc[df['class'] == 1]) >= len(df.loc[df['class'] == 0]):
                return 1
            else:
                return 0
        else:
            left_branch = dectree(df.loc[df[feature] == value], dict, depth)  # Recursion Left
        if len(df.loc[df[feature] != value]) == 0:
            if len(df.loc[df['class'] == 1]) >= len(df.loc[df['class'] == 0]):
                return 1
            else:
                return 0
        else:
            right_branch = dectree(df.loc[df[feature] != value], dict, depth)  # Recursion Right
        l_tuple = (feature, value, True)
        r_tuple = (feature, value, False)
        node_dict = {l_tuple: left_branch, r_tuple: right_branch}
        return node_dict

def id3(x, y, attribute_value=None, depth=0, max_depth=1):
    try: x.insert(loc=0, column='class', value=y)
    except: pass
    depth = max_depth
    columns = list(x)
    del (columns[0])
    dict_all = {}
    for c in columns:
        col = pd.DataFrame((x[c].unique()))
        dict_all = partition(x)
    return dectree(x, dict_all, depth)

def predict_example(x, tree):
    all_keys = list(tree.keys())
    attribute = all_keys[0][0]
    value = all_keys[0][1]
    subtree = None
    node = None
    if x[attribute] == value:
        node = (attribute, value, True)
    else:
        node = (attribute, value, False)
    subtree = tree.get(node)
    # print(node)
    if subtree == 1:
        return 1
    elif subtree == 0:
        return 0
    else:
        return predict_example(x, subtree)

def compute_error(y_true, y_pred):
    correct = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            correct += 1
    return correct

def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

def get_confusion_matrix(y_true, y_pred):
    TP = 0; TN = 0; FN = 0; FP = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1: TP += 1
        elif y_true[i] == 0 and y_pred[i] == 0: TN += 1
        elif y_true[i] == 1 and y_pred[i] == 0: FN += 1
        elif y_true[i] == 0 and y_pred[i] == 1: FP += 1
    # print(TN, TP, FN, FP)
    return [[TP,FN],[FP,TN]]

def main():
    # Monks Dataset
    # df = pd.read_csv("./monks-1.train", header=None)
    # y_train = df[0]
    # x_train = df.drop(0, 1)
    #
    # df_tst = pd.read_csv("./monks-1.test", header=None)
    # y_test = df_tst[0]
    # x_test = df_tst.drop(0, 1)

    # Another Dataset from UCI
    df = pd.read_csv("./train_set.csv", header=None)
    class_index = len(df.columns)-1
    y_train = df[class_index].drop(0).values
    y_train = [int(i) for i in y_train]
    x_train = df.drop(class_index,1).drop(0)

    df_tst = pd.read_csv("./test_set.csv", header=None)
    y_test = df_tst[class_index].drop(0).values
    y_test = [int(i) for i in y_test]
    x_test = df_tst.drop(class_index,1).drop(0)

    # Implemented Decision Tree
    # total_error = 0
    # y_data = y_test
    # x_data = x_test
    # for i in range(1, 3):
    #     tree = id3(x_train, y_train, max_depth=i)
    #     y_pred = []
    #     print("\n")
    #     for example_index in range(len(x_data)): y_pred.append(predict_example(x_data.iloc[example_index], tree))
    #     accuracy = compute_error(y_data, y_pred) / len(y_data)
    #     c_matrix = get_confusion_matrix(y_data, y_pred)
    #     error = (c_matrix[0][1] + c_matrix[1][0]) / len(y_data)
    #     total_error += error
    #     print(tree)
    #     print("======= Depth:", i, " =======")
    #     print("TP: ", c_matrix[0][0], " FN: ", c_matrix[0][1])
    #     print("FP: ", c_matrix[1][0], " TN: ", c_matrix[1][1])
    #     print("=========================\n")
    #     # print("Accuracy: ", round(accuracy*100,2))
    # print("Average Error: ", round((total_error / 10) * 100, 2))

    # Using Scikit Learn
    clf = sk_tree.DecisionTreeClassifier(criterion='entropy')
    classifier = clf.fit(x_train, y_train)
    disp = confusion_matrix(classifier.predict(x_test), y_test)
    print("\n",disp)
    graph_data = sk_tree.export_graphviz(classifier, out_file=None)
    graph = graphviz.Source(graph_data)
    graph.render("adult_salary_scikit_learn")

if __name__ == "__main__":
    main()

