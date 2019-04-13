"""
Author      : Final ProjectTeam 8

Class       : HMC CS 158
Date        : 2019 Jan 17
Description : Music Genre Predictor - Desicion Tree Classifier
"""

import math
import csv
import pandas as pd

# scikit-learn libraties
from sklearn import metrics
from sklearn import tree
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

def performance(y_true, y_pred, metric="accuracy") :
    """
    Calculates the performance metric based on the agreement between the 
    true labels and the predicted labels.
    
    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1_score', 'auroc', 'precision',
                           'sensitivity', 'specificity'        
    
    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1 # map points of hyperplane to +1
    
    # part 2a: compute classifier performance
    if metric == "accuracy":
        score = metrics.accuracy_score(y_true, y_label)

    elif metric == "f1_score":
        score = metrics.f1_score(y_true, y_label)

    elif metric == "auroc":
        score = metrics.roc_auc_score(y_true, y_pred)

    elif metric == "precision":
        score = metrics.precision_score(y_true, y_label)

    elif metric =="sensitivity":
        score = metrics.recall_score(y_true, y_label)

    elif metric == "specificity":
        tn, fp, fn, tp = \
            metrics.confusion_matrix(y_true, y_label).ravel()
        score = float(tn) / (tn +fp)

    else:
        raise Exception(
            str(metric)+" is not a supported performance metric")

    return score

def cv_performance(clf, X, y, kf, metrics=["accuracy"]) :
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.
    
    Parameters
    --------------------
        clf     -- classifier (instance of SVC)
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf      -- model_selection.KFold or model_selection.StratifiedKFold
        metrics -- list of m strings, metrics
    
    Returns
    --------------------
        scores  -- numpy array of shape (m,), average CV performance for each metric
    """

    k = kf.get_n_splits(X, y)
    m = len(metrics)
    scores = np.empty((m, k))
    m=0
    for k, (train, test) in enumerate(kf.split(X, y)) :
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]
        clf.fit(X_train, y_train)
        # use Decision_tree_classifier.predict to make predictions
        y_pred = clf.predict(X_test)
        score = clf.score(X_test, y_test)
        scores[m,k] = score
            
    return scores.mean(axis=1) # average across columns

def select_params(X, y, kf, metrics=["accuracy"]) :
    """
    Sweeps different settings for the hyperparameters of a Decision Tree classifier,
    calculating the k-fold CV performance for each setting and metric,
    then selects the hyperparameters that maximize the average performance for each metric.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf      -- model_selection.KFold or model_selection.StratifiedKFold
        metrics -- list of m strings, metrics
    
    Returns
    --------------------
        params  -- list of m tuples, optimal hyperparameters (max_depth,min_samples) for each metric
    """

    # part 4b: for each metric, select optimal hyperparameters using cross-validation
    
    # create grid of hyperparameters
    # hint: use a small 2x2 grid of hyperparameters for debugging
    depth_range = range(5,21)
    min_samples_range = range(1,15)
    scores = np.empty((len(metrics), len(depth_range), len(min_samples_range)))

    # compute CV scores using cv_performance(...)
    for depth_ind, max_depth in enumerate(depth_range):
        for samples_ind, min_samples in enumerate(min_samples_range):
            print(max_depth,min_samples)
            clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, min_samples_leaf=min_samples) 
            # compute CV scores using cv_performance(...)
            score = cv_performance(clf, X, y, kf, metrics)
            scores[:,depth_ind,samples_ind] = score
    print(scores)
    
    # get best hyperparameters
    best_params = []
    for met_ind, metric in enumerate(scores):
        print (metrics[met_ind])
        print ("maximum score is", metric.max())
        depth_ind, samples_ind = np.unravel_index(metric.argmax(), metric.shape)
        params = (depth_range[depth_ind], min_samples_range[samples_ind])
        print( "max_depth, min_samples=", params)
        best_params.append(params)
    
    return best_params


######################################################################
# main
######################################################################
def main():
    # load dataset
    df = pd.read_csv("../final.csv")
    df = df.drop(columns="Unnamed: 0")
    df = df[((df["Genre"] == "pop") | (df["Genre"] ==  "country"))]
    genres = df["Genre"].unique()

    x_cols    = ["Is_Exp","Danceability","Energy","Key","Loudness","Mode",
        "Speechiness","Acousticness","Instrumentalness","Liveness",
        "Valence","Tempo","Time_Signature"]
    y_cols    = ["Genre"]
    meta_cols = ["Id","Popularity","Name","Artist"]
    X,y,meta = df[x_cols],df[y_cols].iloc[:,0],df[meta_cols]
    y_dict={"pop":0,"country":1}
    y=np.array([y_dict[x] for x in y])

    # Scale the data
    scaler   = StandardScaler()
    scaled_X = scaler.fit_transform(X)

    # split data into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(scaled_X,y, test_size=.2, random_state=1234, stratify=y)

    # optimize parameters with cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
    opt_max_depth, opt_min_samples = select_params(X_train, y_train, skf)[0]
    
    # train classifier
    DTree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=opt_max_depth, min_samples_leaf=opt_min_samples)
    DTree.fit(X_train,y_train)

    # predict genres of test data
    accuracy = DTree.score(X_test,y_test)

    print("Test accuracy of the DTree is")
    print(accuracy)
    print("=============================")

    # compare to stratified dummy classifier
    dummy = DummyClassifier(strategy='stratified')
    dummy.fit(X_train,y_train)
    dummy_accuracy = dummy.score(X_test,y_test)
    print( "Dummy classifier accuracy is" )
    print(dummy_accuracy)


if __name__ == '__main__':
    main()
