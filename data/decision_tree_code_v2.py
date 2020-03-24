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



class decT:

    def partition(self, x):
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

    def entropy(self, y):
        if len(y) == 0:
            return 0
        else:
            p0 = len([c for c in y if c == 0])/len(y)
            p1 = 1 - p0
            if p0 == 0.0 or p0 == 1.0:
                return 0
            else:
                return -((p0 * math.log(p0, 2)) + (p1 * math.log(p1, 2)))


    def mutual_information(self,x,y):
        e_root = self.entropy(y['class'])
        y_true = y[y.index.isin(x)]
        y_false = y[~y.index.isin(x)]
        e_true = self.entropy(y_true['class'])
        e_false = self.entropy(y_false['class'])
        e_tot = (len(y_true) / len(y)) * e_true + (len(y_false) / len(y)) * e_false
        print(e_true,"   ",e_false)
        return e_root - e_tot

    def dectree(self, df, dict, depth):

        if len(dict.keys()) == 0 or len(df['class'].unique()) == 1 or depth == 0:
            try: return st.mode(df['class'])
            except: return 1
        else:
            print("----------------------")
            mutinfo={}
            for k in dict.keys():
                node = dict[k]
                print(k)
                mutinfo[k] = self.mutual_information(node, df)
            splitnode = ()
            Gain=0
            for k in mutinfo.keys():
                if (mutinfo[k]) >= Gain:
                    Gain = mutinfo[k]
                    splitnode = k
            print("\n", splitnode)
            feature = splitnode[0]
            value = splitnode[1]
            dict.pop(splitnode)
            depth -= 1
            left_branch = self.dectree(df.loc[df[feature] == value], dict, depth)  # Recursion Left
            right_branch = self.dectree(df.loc[df[feature] != value], dict, depth)  # Recursion Right
            l_tuple = (feature, value, True)
            r_tuple = (feature, value, False)
            node_dict = {l_tuple: left_branch, r_tuple: right_branch}
            return node_dict

    def id3(self, x,y,attribute_value=None,depth=0,max_depth=5):
        x.insert(loc=0, column='class', value=y)
        depth = max_depth
        columns = list(x)
        del (columns[0])
        dict_all = {}
        for c in columns:
            col = pd.DataFrame((x[c].unique()))
            dict_all = self.partition(x)
        return self.dectree(x, dict_all, depth)

def predict_example(x, tree):
    all_keys = list(tree.keys())
    attribute = all_keys[0][0]
    value = all_keys[0][1]
    subtree = None
    node = None
    if x[attribute] == value: node = (attribute, value, True)
    else: node = (attribute, value, False)
    subtree = tree.get(node)
    print(node)
    if subtree == 1: return 1
    elif subtree == 0: return 0
    else: return predict_example(x, subtree)

def compute_error(y_true, y_pred):
    correct = 0
    for i in range(len(y_pred)):
        # print("Y_Pred: ", y_pred[i],"Y_True: ", y_true[i])
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

def main():
    df = pd.read_csv("./monks-1.train", header=None)
    y = df[0]
    x = df.drop(0, 1)

    df_tst = pd.read_csv("./monks-1.test", header=None)
    y_tst = df_tst[0]
    x_tst = df_tst.drop(0, 1)

    obj = decT()
    tree = obj.id3(x=x,y=y,max_depth=2)
    # visualize(tree)
    print(tree)

    # print(df_tst)

    # print("-------------")
    # predict_example(x_tst.iloc[2], tree)
    # print("-------------")
    # predict_example(x_tst.iloc[10], tree)
    # print("-------------")
    # predict_example(x_tst.iloc[97], tree)
    # print(y_tst[97])


    # y_pred = []
    # for i in range(len(x_tst)):
    #     y_pred.append(predict_example(x_tst.iloc[i], tree))
    #
    # print(len(y_pred))
    # print(len(df_tst))
    # df_tst.insert(loc=0, column='pred', value=y_pred)
    #
    #
    # result = compute_error(y_tst, y_pred)
    # print((result/len(y_pred))*100)


if __name__ == "__main__":
    main()

#
# def compute_error(y_true, y_pred):
#     """
#     Computes the average error between the true labels (y_true) and the predicted labels (y_pred)
#
#     Returns the error = (1/n) * sum(y_true != y_pred)
#     """
#
#     # INSERT YOUR CODE HERE
#     raise Exception('Function not yet implemented!')


# def visualize(tree, depth=0):
#     """
#     Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
#     print the raw nested dictionary representation.
#     DO NOT MODIFY THIS FUNCTION!
#     """
#     print('Visualise function')
#     if depth == 0:
#         print('TREE')
#
#     for index, split_criterion in enumerate(tree):
#         sub_trees = tree[split_criterion]
#
#         # Print the current node: split criterion
#         print('****')
#         print('|\t' * depth, end='')
#         print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))
#
#         # Print the children
#         if type(sub_trees) is dict:
#             visualize(sub_trees, depth + 1)
#         else:
#             print('|\t' * (depth + 1), end='')
#             print('+-- [LABEL = {0}]'.format(sub_trees))


# if __name__ == '__main__':
#     # Load the training data
#     M = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
#     ytrn = M[:, 0]
#     Xtrn = M[:, 1:]
#
#     # Load the test data
#     M = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
#     ytst = M[:, 0]
#     Xtst = M[:, 1:]
#
#     print("Data loaded.")
#
#     # Learn a decision tree of depth 3
#     attr_value_pairs = []
#     for example in Xtrn:
#         for i, value in enumerate(example):
#             if (i, value) not in attr_value_pairs:
#                 attr_value_pairs.append((i, value))
#     attr_value_pairs.sort()
#     print(attr_value_pairs)
#
#     # print(ytrn)
#     decision_tree = id3(Xtrn, ytrn, attr_value_pairs, max_depth=3)
#     visualize(decision_tree)
#     # print(decision_tree)
#
#     # Compute the test error
# #    y_pred = [predict_example(x, decision_tree) for x in Xtst]
# #    tst_err = compute_error(ytst, y_pred)
# #
# #    print('Test Error = {0:4.2f}%.'.format(tst_err * 100))
