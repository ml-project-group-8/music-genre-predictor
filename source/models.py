from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn import tree

import warnings; warnings.simplefilter('ignore')

def cv_performance(clf, X, y, kf, metrics=["accuracy"]) :
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation accuracy for classifier
    by averaging the performance across folds.

    Adapted for HW6
    """
    y=np.array(y.tolist())
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

    Adapted from HW6
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
            clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, min_samples_leaf=min_samples)
            # compute CV scores using cv_performance(...)
            score = cv_performance(clf, X, y, kf, metrics)
            scores[:,depth_ind,samples_ind] = score

    # get best hyperparameters
    best_params = []
    for met_ind, metric in enumerate(scores):
        depth_ind, samples_ind = np.unravel_index(metric.argmax(), metric.shape)
        params = (depth_range[depth_ind], min_samples_range[samples_ind])

        best_params.append(params)

    return best_params
class Models:

    def __init__(self):

        df     = pd.read_csv("data/lyrical_genius.csv")
        df = df[((df["Genre"] == "pop") | (df["Genre"] ==  "country"))]
        df = df.drop(columns="Unnamed: 0")
        df = df.drop_duplicates(subset=["Name","Artist"],keep=False)

        x_cols    = ["Is_Exp","Danceability","Energy","Key","Loudness","Mode","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo","Time_Signature"]
        y_cols    = ["Genre"]
        meta_cols = ["Id","Popularity","Name","Artist"]

        X,y,meta  = df[x_cols],df[y_cols].iloc[:,0],df[meta_cols]

        scaler   = StandardScaler()
        scaled_X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X,y, test_size=.2, random_state=1234, stratify=y)

        # KNN
        knn = KNeighborsClassifier()
        #create a dictionary of all values we want to test for n_neighbors
        param_grid = {'n_neighbors': np.arange(1,25)}
        #use gridsearch to test all values for n_neighbors
        knn_gscv = GridSearchCV(knn, param_grid, cv=5)
        #fit model to data
        knn.fit(X_train, y_train)

        # Logistic
        logclf = LogisticRegression(solver="lbfgs", multi_class="ovr")
        logclf.fit(X_train,y_train)


        # DTree
        # optimize parameters with cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
        opt_max_depth, opt_min_samples = select_params(X_train, y_train, skf)[0]

        # train classifier
        DTree = tree.DecisionTreeClassifier(criterion="entropy", max_depth=opt_max_depth, min_samples_leaf=opt_min_samples)
        DTree.fit(X_train,y_train)

        # Dummy
        dummy = DummyClassifier(strategy='stratified')
        dummy.fit(X_train,y_train)

        self.scaler = scaler
        self.lyrics = df["Lyrics"]
        self.y      = y
        self.X      = scaled_X
        self.df     = df
        self.Xdef   = X
        self.meta   = meta

        self.xtrain = X_train
        self.ytrain = y_train
        self.xtest  = X_test
        self.ytest  = y_test

        self.models = {
            "KNN": knn,
            "Logistic": logclf,
            "DesicionTree": DTree,
            "Dummy": dummy
        }
    def scale(self,newx):
        order = list(self.Xdef)
        x_data = np.asarray([newx[x] for x in order]).reshape(1,-1)
        return self.scaler.transform(x_data)

    def predict(self,newx):
        scaled = self.scale(newx)

        ret = {}
        for model in self.models:
            ret[model] = list(self.models[model].predict(scaled))[0].title()
        return ret
