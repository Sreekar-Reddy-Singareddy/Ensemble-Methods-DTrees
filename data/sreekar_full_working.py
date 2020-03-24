import numpy as np
import copy
import math

attr_value_pair_reference = None

def entropy(y):
    if len(y) == 0: return 0

    y_based_split = partition(y)
    y_yes = y_based_split.get(1); #print(len(y_yes))
    y_no = y_based_split.get(0); #print(len(y_no))
    p_yes = len(y_yes)/len(y)
    p_no = len(y_no)/len(y)
    A = B = 0
    if p_yes == 0:
        A = 0
    else:
        A = -(p_yes * math.log(p_yes, 2))
    if p_no == 0:
        B = 0
    else:
        B = -(p_no * math.log(p_no, 2))
    return (A+B)

def mutual_information(x, y):
    entropy_y_before_split = entropy(y) # E(S)
    x_based_split = partition(x)
    x_yes = x_based_split.get(1) # Indices of dataset where x == v
    x_no = x_based_split.get(0) # Indices of dataset where x != v
    x_yes_ratio = len(x_yes)/len(x)
    x_no_ratio = len(x_no)/len(x)
    y_where_x_yes = []
    y_where_x_no = []
    for index in x_yes: y_where_x_yes.append(y[index])
    for index in x_no: y_where_x_no.append(y[index])
    entropy_y_where_x_yes = entropy(y_where_x_yes) # E(Sy)
    entropy_y_where_x_no = entropy(y_where_x_no) # E(Sn)
    mutual_info = entropy_y_before_split - (x_yes_ratio * entropy_y_where_x_yes) - (x_no_ratio * entropy_y_where_x_no)
    # print("X True: ", x_yes_ratio, " X False: ", x_no_ratio)
    print("E(S) = ", entropy_y_before_split," E(Sy) = ", entropy_y_where_x_yes, " E(Sn) = ",entropy_y_where_x_no,"MI = ", mutual_info)
    return mutual_info

def partition(x):
    yes = []; no = []
    for i,row in enumerate(x):
        if row == 1:
            yes.append(i)
        else:
            no.append(i)
    return {1: yes, 0: no}

def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    class_frequency = compute_class_frequency(y)

    # Check boundary conditions to STOP
    if len(attribute_value_pairs) == 0 or depth == max_depth:
        if class_frequency[0] > class_frequency[1]: return 0
        else: return 1
    elif class_frequency[0] == len(y): return 0
    elif class_frequency[1] == len(y): return 1

    else:
        # Mutual Information for all the attr-value pairs
        all_mutual_infos = []
        for i, pair in enumerate(attribute_value_pairs):
            print(pair)
            pair_mut_info = mutual_information([row[attr_value_pair_reference.index(pair)] for row in x] ,y)
            all_mutual_infos.append(pair_mut_info)

        # Get the pair with maximum mutual information
        max_mutual_info = max(all_mutual_infos)
        max_mutual_index = all_mutual_infos.index(max_mutual_info)
        max_pair = attribute_value_pairs[max_mutual_index]
        # print("Max MI: ", max_mutual_info, " Index: ", max_mutual_index, " Pair: ", max_pair," Pairs: ", len(attribute_value_pairs))

        # Get the feature and value of that pair
        split_criteria = attr_value_pair_reference.index(max_pair)

        # Split data to left and right subsets
        data_after_split = split_data(split_criteria, x, y)

        # Remove the attr-pair from list
        attribute_value_pairs.remove(max_pair)
        l_attr_pairs = copy.copy(attribute_value_pairs)
        r_attr_pairs = copy.copy(attribute_value_pairs)
        depth += 1
        left_branch = id3(data_after_split[0], data_after_split[1], l_attr_pairs, depth, max_depth) # Recursion Left
        right_branch = id3(data_after_split[2], data_after_split[3], r_attr_pairs, depth, max_depth) # Recursion Right

        # Node description for building the tree
        l_tuple = (max_pair[0], max_pair[1], True)
        r_tuple = (max_pair[0], max_pair[1], False)
        node_dict = {l_tuple: left_branch, r_tuple: right_branch}
        return node_dict

def split_data(attribute, x, y):
    xL = []; xR = []; yL = []; yR = []
    for i,example in enumerate(x):
        if example[attribute] == 1:
            xL.append(x[i])
            yL.append(y[i])
        else:
            xR.append(x[i])
            yR.append(y[i])
    print("XL: ", len(xL)," XR: ", len(xR))
    return [xL,yL,xR,yR]

def compute_class_frequency(y):
    class_1 = class_0 = 0
    for example in y:
        if example == 1:
            class_1 += 1
        else:
            class_0 += 1
    return [class_0, class_1]

def predict_example(x, tree):
    all_keys = list(tree.keys())
    attribute = all_keys[0][0]
    value = all_keys[0][1]
    subtree = None
    node = None
    if x[attribute] == value: node = (attribute, value, True)
    else: node = (attribute, value, False)
    subtree = tree.get(node)
    # print(node)
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
    print('Visualise function')
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('****')
        # print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            # print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))

if __name__ == '__main__':

    # Load the training data
    M = np.genfromtxt('./monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    # Process the list of all possible attr-value pairs
    attr_value_pairs = []
    for example in Xtrn:
        for i, value in enumerate(example):
            if (i,value) not in attr_value_pairs:
                attr_value_pairs.append((i,value))

    # List of attr-value pairs
    attr_value_pairs.sort()
    attr_value_pair_reference = copy.copy(attr_value_pairs)

    # Preprocessing the dataset based on attr-value pairs
    Xtrn_new = []
    for i,row in enumerate(Xtrn):
        temp_row = []
        for j,pair in enumerate(attr_value_pairs):
            if pair[1] == row[pair[0]]:
                temp_row.append(1)
            else:
                temp_row.append(0)
        Xtrn_new.append(temp_row)
        # print(temp_row)

    # Train the ID3 model
    decision_tree = id3(Xtrn_new, ytrn, attr_value_pairs, max_depth=70)
    print(decision_tree)

    # Compute the error using training set
    # y_pred = []
    # for row in Xtrn:
    #     # print("===========================")
    #     y_pred.append(predict_example(row, decision_tree))
    #
    # tst_err = compute_error(ytrn, y_pred)
    # print("Train: ", len(ytrn), " Train_Pred: ", len(y_pred))
    # print(tst_err/len(y_pred))

    # Compute the error using test set
    y_pred = []
    for row in Xtst:
        # print("===========================")
        y_pred.append(predict_example(row, decision_tree))

    tst_err = compute_error(ytst, y_pred)
    print("Test: ", len(ytst), " Pred: ", len(y_pred))
    print(tst_err / len(y_pred))