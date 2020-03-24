import numpy as np
import statistics as st
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import tree as sk_tree
from sklearn.metrics import confusion_matrix
import graphviz
import sys
import random

# Partition Function:
# -receives the dataframe and creates a dictionary of attribute-value pairs
# - it is called once for creation of starting dictionary which is passed to the ID3 function
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

# Entropy function:It receives the class variable as input, and does the following checks
# if the size of data =0 (nothing to calculate entropy upon), returns 0
# if proportion of one class in a split subset is 0, return 0 for that entropy ( homogenous class)
# else calculate the entropy  - Summation(Pi*log(Pi) (i=0,1)
# returns value to the mutual_information function
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

# Mutual information function :
# 1.receives the indices which corresponds to the attribute-value pair presently considered
# 2. receives the present subset of the data, on which entropy and mutual information is to be evaluated
# 3. receives entropy before and after  binary split, calculates weighted entropy after split
# 4. finally subtracts before - weighted entropy to return mutual information

def mutual_information(x, y):
    e_root = entropy(y['class'])
    y_true = y[y.index.isin(x)]
    y_false = y[~y.index.isin(x)]
    e_true = entropy(y_true['class'])
    e_false = entropy(y_false['class'])
    e_tot = (len(y_true) / len(y)) * e_true + (len(y_false) / len(y)) * e_false
    return e_root - e_tot

# dectree: this is main binary decision tree recursion implementation code
# terminal conditions are following ID3 algorithm:
#     -if no splitting attribute left, return mode (max frequency) class the left over data (1 or 0)
#     -if left over data is of only one class, i.e. homogenous , return that class value
#     -if max depth has reached, return mode  class the left over data
#     -if no data is left to be split, return prior to split mode of class ( don't recursively call anymore)
# final output is a nested dictionary representing the decision tree

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

# ID3 function -this is the controlling function for the decision tree
# 1. It receives the data , class/label column , and max depth from user main function
# the depth column is given in skeleton code, however used a bit differently in present context:
#   -as we are not making ID3 recursively execute, we just pass these arguments to our recursive function (dectree)
# this function calls the partition function to create the dictionary of attribute value pairs
# it calls the main decision tree function (dectree) with all required arguments
# receives and returns the decision tree (nested dictionary) created to main function

def id3(x, y, attribute_value=None, depth=0, max_depth=1):
    # x.insert(loc=0, column='class', value=y)
    try:
        x.insert(loc=0, column='class', value=y)
    except:
        pass
    depth = max_depth
    columns = list(x)
    del (columns[0])
    dict_all = {}
    for c in columns:
        col = pd.DataFrame((x[c].unique()))
        dict_all = partition(x)
    return dectree(x, dict_all, depth)

# predict_example function: It takes in one example at a time and traverses through the binary decision tree to
#                          reach the end node. Based on the end node reached, the mode class of the node is the
#                          predicted class of this example. It returns the predicted class to main function

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
    # print(node)   # uncomment this line to see the traversed nodes by the example
    if subtree == 1:
        return 1
    elif subtree == 0:
        return 0
    else:
        return predict_example(x, subtree)

# compute_error function : takes in predicted and actual class columns and computes the mis-classification proportion
def compute_error(y_true, y_pred):
    correct = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            correct += 1
    return (1 - correct / len(y_true))

# visualize function: provided visualization function to display the decision tree in tree structure
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

# get_confusion_matrix function : returns the cross tab of actual vs predicted class frequencies
def get_confusion_matrix(y_true, y_pred):
    TP = 0;
    TN = 0;
    FN = 0;
    FP = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
    # print(TN, TP, FN, FP)
    return [[TP, FN], [FP, TN]]

# This method takes main training data set and returns the ensemble hypotheses
# as an array.
def bagging (x, max_depth, num_trees, class_column=1):
    bootstraps = create_bootstraps(x,num_trees,class_column=class_column)
    all_trees = []
    for bootstrap in bootstraps:
        tree_b = id3(bootstrap[0], bootstrap[1], max_depth=max_depth)
        weight_b = 1 # This is set as 1 because all trees are independent in bagging
        all_trees.append((weight_b, tree_b))
    return all_trees

# This method takes main training data and creates "k" bootstrap samples
# each of size same as that of main training data.
# Returns these bootstraps as an array of tuples.
# Each tuple format is (x_data, y_data)
def create_bootstraps (df, k, class_column=1):
    # Handle the simple errors
    if (class_column < 1):
        print("Invalid class column.")
        return
    if (k < 1):
        print("Invalid number of bootstraps. Must be >= 1.")
        return

    # Create k bootstrap samples
    bootstraps = [] # Empty bins
    class_column-=1
    for i in range(0, k):
        df_k = df.sample(frac=1, replace=True)
        y_k = df_k[class_column]
        x_k = df_k.drop(class_column, 1)
        bootstraps.append((x_k, y_k))
        # print(df_k.head())
        # x_k = pd.DataFrame(columns=list(x.columns)); y_k = pd.Series();
        # for n in range(0, dataset_size):
            # rand_index = random.randint(0, dataset_size-1)
            # rand_x = x.iloc[rand_index]
            # rand_y = pd.Series([y.iloc[rand_index]])
            # x_k = x_k.append(rand_x, ignore_index=True)
            # y_k = y_k.append(rand_y, ignore_index=True)
    return bootstraps

# Takes the test set and computes the prediction.
# Returns the array of BEST predictions as combined
# by the ensemble hypothesis.
# For ensemble models, "h_ens" parameter must be passed
# For a single tree model, "h_tree" parameter must be passed.
def predict_test_set(test_x, type, h_ens=[], h_tree=None):
    if type != 'tree' and type != 'bagging_tree' and type != 'boosting_tree':
        print("Provide the type of model - 'tree', 'bagging_tree', or 'boosting_tree'")
        return
    num_of_examples = len(list(test_x.index))
    predictions = []
    for i in range(0, num_of_examples):
        preds_i = []
        for h in h_ens:
            preds_i.append(h[0] * predict_example(test_x.iloc[i], h[1]))
        if (type == "bagging_tree"):
            try :predictions.append(st.mode(preds_i)) # Final prediction of bagging
            except: predictions.append(1) # Tie breaking
        elif (type == "boosting_tree"): pass
        elif (type == "tree"): predictions.append(predict_example(test_x.iloc[i], h_tree)) # Prediction using simple tree
    return predictions

# main function which reads file, splits the train and test data in feature data (x_...) and class column (y_...)
# call id3 function with multiple max depth (running in loop).
# calls the predict_example function on the test data and created decision tree
# calls compute_error to get the test error
# calls get_confusion_matrix to get confusion matrix and calculate accuracy
# reports the results
def main():
    ##*******--Reading part (start to end) for Problem parts (a),(b),(c)---*******# Comment out when uncommenting reading part for part (d) below
    ## start
    trial_data_name = "monks-1"
    try:
        trial_data_name = sys.argv[1]  # use this line while triggering python script from command line ,
                                   # pass data name (without the extension) as argument
        if len(trial_data_name) == 0:
            trial_data_name = "monks-1"  # please change this name to get the output file labelled with dataset name
    except:
        pass

    df = pd.read_csv("./{}.train".format(trial_data_name), header=None)
    y_train = df[0]
    x_train = df.drop(1, 1)
    print(x_train.head())

    # df_tst = pd.read_csv("./{}.test".format(trial_data_name), header=None)  # to get train error
    # # read the train file instead of test file
    # y_test = df_tst[0]
    # x_test = df_tst.drop(0, 1)
    # ## end
    #
    # # #*******--Reading part for Problem parts (d)---*******# - Own preprocessed data (source: UCI repository) : uncomment from start to end
    # ## start
    # trial_data_name = "Adult_Income"  # please change this name to get the error file labelled with data name
    # # df = pd.read_csv("C:\\MS CS\\Spring 2020\\ML\\PA\\SXC190070_PA1\\UCI_AdultIncome_Data\\train_set.csv")
    # # y_train = df["Income_class"]
    # # x_train = df.drop("Income_class", 1)
    # # df_tst = pd.read_csv("C:\\MS CS\\Spring 2020\\ML\\PA\\SXC190070_PA1\\UCI_AdultIncome_Data\\test_set.csv")
    # # y_test = df_tst["Income_class"]
    # # x_test = df_tst.drop("Income_class", 1)
    # ## end
    #
    # # Implemented Decision Tree - For Test error
    # print("\n Running Decision Tree and evaluating results on Test data\n")
    # total_error = 0
    # y_data = y_test
    # x_data = x_test
    # error_dict = {}
    # depth_range = 6;
    # for i in range(1, depth_range):
    #     tree = id3(x_train, y_train, max_depth=i)
    #     y_pred = []
    #     print("\n")
    #     for example_index in range(len(x_data)): y_pred.append(predict_example(x_data.iloc[example_index], tree))
    #     error = compute_error(y_data, y_pred)
    #     error_dict[i] = round(error * 100, 2)
    #     c_matrix = get_confusion_matrix(y_data, y_pred)
    #     accuracy = (c_matrix[0][0] + c_matrix[1][1]) / (
    #             c_matrix[0][0] + c_matrix[0][1] + c_matrix[1][0] + c_matrix[1][1])
    #     total_error += error
    #     print("===================== Depth:", i, " =====================")
    #     print("Decision Tree: ", tree)
    #     print("\n\n")
    #     visualize(tree)
    #     print("TP: ", c_matrix[0][0], " FN: ", c_matrix[0][1])
    #     print("FP: ", c_matrix[1][0], " TN: ", c_matrix[1][1])
    #     print("=========================\n")
    #     print("Accuracy: ", round(accuracy * 100, 2))
    #     print("Error: ", round(error * 100, 2))
    # print("Average Error: ", round((total_error / depth_range) * 100, 2))
    # error_output = "error_file_test_{}.csv".format(trial_data_name)
    # (pd.DataFrame.from_dict(data=error_dict, orient='index').to_csv(error_output, header=False))

    # ## Implemented Decision Tree - For Train error
    # print("****************************************************************")
    # print("\n Running Decision Tree and evaluating results on Train data\n")
    # total_error = 0
    # y_data = y_train
    # x_data = x_train
    # error_dict = {}
    # for i in range(1, 3):
    #     tree = id3(x_train, y_train, max_depth=i)
    #     y_pred = []
    #     print("\n")
    #     for example_index in range(len(x_data)): y_pred.append(predict_example(x_data.iloc[example_index], tree))
    #     error = compute_error(y_data, y_pred)
    #     error_dict[i] = round(error * 100, 2)
    #     c_matrix = get_confusion_matrix(y_data, y_pred)
    #     accuracy = (c_matrix[0][0] + c_matrix[1][1]) / (
    #             c_matrix[0][0] + c_matrix[0][1] + c_matrix[1][0] + c_matrix[1][1])
    #     total_error += error
    #
    #     print("===================== Depth:", i, " =====================")
    #     print("Decision Tree: ", tree)
    #     print("\n\n")
    #     visualize(tree)
    #     print("TP: ", c_matrix[0][0], " FN: ", c_matrix[0][1])
    #     print("FP: ", c_matrix[1][0], " TN: ", c_matrix[1][1])
    #     print("=========================\n")
    #     print("Accuracy: ", round(accuracy * 100, 2))
    #     print("Error: ", round(error * 100, 2))
    # print("Average Error: ", round((total_error / 10) * 100, 2))
    # error_output = "C:\\MS CS\\Spring 2020\\ML\\PA\\\SXC190070_PA1\\output\\error_file_train_{}.csv".format(trial_data_name)
    # (pd.DataFrame.from_dict(data=error_dict, orient='index').to_csv(error_output, header=False))

    ## Below code is for part(c) -Scikit Learn -Uncomment the below part from start to end ( the reading part is same as
    ## for part (a) and part (b)
    ## start
    # print("Below is the scikit learn decision tree output for {} data".format(trial_data_name))
    # clf = sk_tree.DecisionTreeClassifier()
    # classifier = clf.fit(x_train, y_train)
    # disp = confusion_matrix(classifier.predict(x_test), y_test)
    # print("\n",disp)
    # graph_data = sk_tree.export_graphviz(classifier, out_file=None)
    # graph = graphviz.Source(graph_data)
    # graph.render("C:\\MS CS\\Spring 2020\\ML\\PA\\\SXC190070_PA1\\output\\decision_tree1")
    ## end

# The main execution of the assignment begins here
def pro_assign_2():
    data_set_name = "monks-1"
    data_columns_to_drop = []
    data_class_column = 1

    # Read the data files
    train_data_path = "./{}.train".format(data_set_name)
    test_data_path = "./{}.test".format(data_set_name)
    train_df = pd.read_csv(train_data_path, delimiter = ",", header=None)
    test_df = pd.read_csv(test_data_path, delimiter = ",", header=None)

    # Drop the unwanted columns
    for c in data_columns_to_drop:
        del train_df[c-1]
        del test_df[c-1]

    # Extract the class column
    train_y = train_df[data_class_column-1] #  Bruises column data is stored here - Train
    del train_df[data_class_column-1]
    test_y = test_df[data_class_column-1] #  Bruises column data is stored here - Test
    del test_df[data_class_column-1]

    # Send the training data to the bagging algorithm
    bagging(train_df, train_y, 3, 10)

# The main execution of the assignment begins here
def pro_assign_2_auto(depths=[], trees=[]):
    data_set_name = "monks-1"
    data_columns_to_drop = []
    data_class_column = 1

    # Read the data files
    train_data_path = "./{}.train".format(data_set_name)
    test_data_path = "./{}.test".format(data_set_name)
    train_df = pd.read_csv(train_data_path, delimiter=",", header=None)
    test_df = pd.read_csv(test_data_path, delimiter=",", header=None)
    test_y = list(test_df[data_class_column-1])
    del test_df[data_class_column - 1]

    # Drop the unwanted columns
    for c in data_columns_to_drop:
        del train_df[c - 1]
        del test_df[c - 1]

    output = open("./output.txt", "w")
    for depth in depths:
        for tree_len in trees:
            # Send the training data to the bagging algorithm
            all_trees = bagging(train_df, class_column=1, max_depth=depth, num_trees=tree_len)

            # Predict the test set with all the trees
            predictions = predict_test_set(test_df, type="bagging_tree", h_ens=all_trees)

            # Compute the error and accuracy
            error = compute_error(test_y, predictions)
            print("Error: ", round(error * 100, 2), file=output)
            print("Accuracy: ", round((1 - error) * 100, 2), file=output)

            # Gets the confusion matrix
            confusion_matrix = get_confusion_matrix(test_y, predictions)
            print("=================== Confuction Matrix ==================", file=output)
            print(confusion_matrix[0], file=output)
            print(confusion_matrix[1], file=output)

    output.close()

if __name__ == "__main__":
    # main()
    # pro_assign_2()
    pro_assign_2_auto(depths=[3,5],trees=[10,20])

