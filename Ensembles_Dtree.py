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
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

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
def entropy(y, dist=None):
    if len(y) == 0:
        return 0
    else:
        p0 = p1 = 0
        if dist is None:
            p0 = len([c for c in y if c == 0]) / len(y)
            p1 = 1 - p0
        else:
            for index in y.index:
                if y[index] == 0:
                    p0 += dist[index]
                else:
                    p1 += dist[index]
            p0 = p0/(p0+p1)
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
def mutual_information(indices, df, hasDist):
    e_root = 0
    if hasDist:
        e_root = entropy(df['class'], dist=df["distribution"])
    else :
        e_root = entropy(df['class'])

    df_true = df[df.index.isin(indices)]
    df_false = df[~df.index.isin(indices)]

    e_true = 0; e_false = 0
    if hasDist:
        e_true = entropy(df_true['class'], dist=df["distribution"])
        e_false = entropy(df_false['class'], dist=df["distribution"])
    else:
        e_true = entropy(df_true['class'])
        e_false = entropy(df_false['class'])

    e_tot = (len(df_true) / len(df)) * e_true + (len(df_false) / len(df)) * e_false
    return abs(e_root - e_tot)

# dectree: this is main binary decision tree recursion implementation code
# terminal conditions are following ID3 algorithm:
#     -if no splitting attribute left, return mode (max frequency) class the left over data (1 or 0)
#     -if left over data is of only one class, i.e. homogenous , return that class value
#     -if max depth has reached, return mode  class the left over data
#     -if no data is left to be split, return prior to split mode of class ( don't recursively call anymore)
# final output is a nested dictionary representing the decision tree
def dectree(df, dict, depth, hasDist):
    if len(dict.keys()) == 0 or len(df['class'].unique()) == 1 or depth == 0:
        if len(df.loc[df['class'] == 1]) >= len(df.loc[df['class'] == 0]):
            return 1
        else:
            return 0
    else:
        mutinfo = {}
        for k in dict.keys():
            node = dict[k]
            mutinfo[k] = mutual_information(node, df, hasDist)
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
            left_branch = dectree(df.loc[df[feature] == value], dict, depth, hasDist)  # Recursion Left
        if len(df.loc[df[feature] != value]) == 0:
            if len(df.loc[df['class'] == 1]) >= len(df.loc[df['class'] == 0]):
                return 1
            else:
                return 0
        else:
            right_branch = dectree(df.loc[df[feature] != value], dict, depth, hasDist)  # Recursion Right
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
def id3(x, y, attribute_value=None, depth=0, max_depth=1, hasDist=False):
    # x.insert(loc=0, column='class', value=y)
    try:
        x.insert(loc=0, column='class', value=y)
    except:
        pass
    depth = max_depth
    columns = list(x)
    del (columns[0])
    del (columns[len(columns)-1])
    dict_all = {}
    for c in columns:
        col = pd.DataFrame((x[c].unique()))
        if hasDist: dict_all = partition(x.drop("distribution",1))
        else: dict_all = partition(x)
    return dectree(x, dict_all, depth, hasDist)

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

# This method accepts train set and num of ensemble trees
# to train. Returns an array of ensembles [(alpha_i, hyp_i)]
def boosting (data, max_depth, num_stumps, class_column=1):
    y = data[class_column-1]
    x = data.drop(class_column-1, 1)
    sample_size = len(list(x.index))
    d_i = [1/sample_size]*sample_size # The initial distribution

    ensembles = [] # Array of ensembles [(alpha_i, hyp_i)]
    for i in range(0,num_stumps):
        x.insert(loc=len(x.columns), column="distribution", value=d_i)
        h_i = id3(x,y,max_depth=max_depth, hasDist=True) # ith decision tree
        d_i = list(x["distribution"])
        del x["distribution"]
        y_pred = predict_test_set(x, type="tree", h_tree=h_i)
        err_i = compute_error_boosting(y, y_pred, d_i) # error of ith decision tree
        alpha_i = get_hypothesis_weight(err_i) # weight of ith decision tree
        d_i = get_new_distribution(d_i, alpha_i, y, y_pred) # new distribution for next dtree
        ensembles.append((alpha_i, h_i))
    return ensembles

def compute_error_boosting (y_true, y_pred, d_i):
    total = 0; error = 0
    for i in range(0, len(y_true)):
        if (y_pred[i] != y_true[i]):
            error += d_i[i]
    return error/sum(d_i)

# This method takes error and returns the alpha value
def get_hypothesis_weight(error):
    a = (1-error)/error
    return 0.5*math.log(a)

# This method computes new distribution based on the current predictions
def get_new_distribution(prev_dis, alpha, y_true, y_pred):
    new_dis = [-1]*len(prev_dis)
    for i in range(0, len(prev_dis)):
        if y_true[i] == y_pred[i]: # Decrease the weight for correct prediction
            new_dis[i] = prev_dis[i]*math.exp(-alpha)
        else: # Increase the weight for incorrect prediction
            new_dis[i] = prev_dis[i]*math.exp(alpha)
    return new_dis

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
            pred = predict_example(test_x.iloc[i], h[1])
            if (type == "boosting_tree" and pred == 0): preds_i.append(h[0] * -1)
            else: preds_i.append(h[0] * pred)
        if (type == "bagging_tree"):
            try :predictions.append(st.mode(preds_i)) # Final prediction of bagging
            except: predictions.append(1) # Tie breaking
        elif (type == "boosting_tree"):
            if sum(preds_i) > 0: predictions.append(1)
            else: predictions.append(0)
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
def pro_assign_2_bagging(depths=[], trees=[]):
    data = read_data("monks-1")
    train_df = data[0]
    test_x = data[1]
    test_y = data[2]

    for depth in depths:
        for tree_len in trees:
            # Send the training data to the bagging algorithm
            all_trees = bagging(train_df, class_column=1, max_depth=depth, num_trees=tree_len)

            # Predict the test set with all the trees
            predictions = predict_test_set(test_x, type="bagging_tree", h_ens=all_trees)
            print_report(predictions, test_y, depth=depth, trees=tree_len)

# The main execution of the assignment begins here
def pro_assign_2_boosting(depths=[], trees=[]):
    data_set_name = "mushroom"
    data_columns_to_drop = []
    data_class_column = 1

    # Read the data files
    train_data_path = "./data/{}.train".format(data_set_name)
    test_data_path = "./data/{}.test".format(data_set_name)
    train_df = pd.read_csv(train_data_path, delimiter=",", header=None)
    test_df = pd.read_csv(test_data_path, delimiter=",", header=None)
    test_y = list(test_df[data_class_column-1])
    del test_df[data_class_column - 1]

    # Drop the unwanted columns
    for c in data_columns_to_drop:
        del train_df[c - 1]
        del test_df[c - 1]

    for depth in depths:
        for tree in trees:
            # Boosting algorithm
            all_trees = boosting(train_df, depth, tree, data_class_column)

            # Predict the test set with all the trees
            predictions = predict_test_set(test_df, type="boosting_tree", h_ens=all_trees)

            # Compute the error and accuracy
            error = compute_error(test_y, predictions)
            print("Error: ", round(error * 100, 2))
            print("Accuracy: ", round((1 - error) * 100, 2))

            # Gets the confusion matrix
            confusion_matrix = get_confusion_matrix(test_y, predictions)
            print("=================== Confuction Matrix ==================")
            print(confusion_matrix[0])
            print(confusion_matrix[1])

# Scikit learn bagging
def scikit_bagging(depths=[], trees=[]):
    data = read_data("monks-1")
    train_df = data[0]
    test_x = data[1]
    test_y = data[2]

    for d in depths:
        for t in trees:
            dtree = DecisionTreeClassifier(max_depth=d)
            clf = BaggingClassifier(base_estimator= dtree,n_estimators=t, bootstrap=True).fit(train_df.drop(0,1), train_df[0])
            pred = clf.predict(test_x)
            print_report(pred, test_y, depth=d, trees=t)

# Scikit learn boosting
def scikit_boosting(depths=[], stumps=[]):
    data = read_data("monks-1")
    train_df = data[0]
    test_x = data[1]
    test_y = data[2]

    for d in depths:
        for s in stumps:
            dtree = DecisionTreeClassifier(max_depth=d)
            clf = AdaBoostClassifier(base_estimator=dtree, n_estimators=s).fit(train_df.drop(0,1), train_df[0])
            pred = clf.predict(test_x)
            print_report(pred, test_y, depth=d, trees=s)

def print_report(y_pred, y_true, depth=0, trees=1):
    err = compute_error(y_true, y_pred)
    acc = 1 - err
    print("========================= Depth: {} and Trees: {} ==============================".format(depth, trees))
    print("Error   : ", round(err * 100, 2))
    print("Accuracy: ", round(acc * 100, 2))
    print("C Matrix: \n", confusion_matrix(y_true, y_pred))
    print("========================= *********************** ==============================".format(depth, trees))

def read_data(data_set_name, data_class_column=1, data_columns_to_drop=[]):
    # Read the data files
    train_data_path = "./data/{}.train".format(data_set_name)
    test_data_path = "./data/{}.test".format(data_set_name)
    train_df = pd.read_csv(train_data_path, delimiter=",", header=None)
    test_df = pd.read_csv(test_data_path, delimiter=",", header=None)
    test_y = list(test_df[data_class_column - 1])
    del test_df[data_class_column - 1]

    # Drop the unwanted columns
    for c in data_columns_to_drop:
        del train_df[c - 1]
        del test_df[c - 1]

    return (train_df, test_df, test_y)

if __name__ == "__main__":
    # main()
    # pro_assign_2()
    # pro_assign_2_bagging(depths=[3,5],trees=[10,20])
    # pro_assign_2_boosting([1,2], [20,40])
    scikit_bagging([3,5], [10,20])
    # scikit_boosting([1,2], [20,40])

