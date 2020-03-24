import numpy as np
import statistics as st
import pandas as pd
import math
from sklearn import tree as sk_tree
from sklearn.metrics import confusion_matrix
import graphviz

if __name__ == "__main__":
    # Monks Dataset
    df = pd.read_csv("./monks-1.train", header=None)
    y_train = df[0]
    x_train = df.drop(0, 1)

    df_tst = pd.read_csv("./monks-1.test", header=None)
    y_test = df_tst[0]
    x_test = df_tst.drop(0, 1)

    # Another Dataset from UCI
    # df = pd.read_csv("./train_set.csv", header=None)
    # class_index = len(df.columns)-1
    # y_train = df[class_index].drop(0).values
    # y_train = [int(i) for i in y_train]
    # x_train = df.drop(class_index,1).drop(0)
    #
    # df_tst = pd.read_csv("./test_set.csv", header=None)
    # y_test = df_tst[class_index].drop(0).values
    # y_test = [int(i) for i in y_test]
    # x_test = df_tst.drop(class_index,1).drop(0)

    # Using Scikit Learn
    clf = sk_tree.DecisionTreeClassifier()
    classifier = clf.fit(x_train, y_train)
    disp = confusion_matrix(classifier.predict(x_test), y_test)
    print("\n",disp)
    graph_data = sk_tree.export_graphviz(classifier, out_file=None)
    graph = graphviz.Source(graph_data)
    graph.render("decision_tree")