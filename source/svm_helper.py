from string import punctuation

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# scikit-learn libraries
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.utils import shuffle

######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy", test=False) :
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
   
    if test:
        y_label = y_pred

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_label, labels=[0,1]).ravel()
    
    if metric == "accuracy":
        if (tp + tn + fp + fn) == 0:
            return 0
        return (tp + tn) / (tp + tn + fp + fn)
    elif metric == "f1_score":
        if (tp + fp + fn) == 0:
            return 0
        return 2.0*(tp) / (2.0 * tp + fp + fn)
    elif metric == "auroc":
        return metrics.roc_auc_score(y_true, y_pred)
    elif metric == "precision":
        if (tp + fp) == 0:
            return 0
        return (tp) / (tp + fp)
    elif metric =="sensitivity":
        if (tp + fn) == 0:
            return 0
        return (tp) / (tp + fn)
    elif metric == "specificity":
        if (tn + fp) == 0:
            return 0
        return (tn) / (tn + fp)

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

    for k, (train, test) in enumerate(kf.split(X, y)) :
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        # use SVC.decision_function to make ``continuous-valued'' predictions
        y_pred = clf.decision_function(X_test)
        for m, metric in enumerate(metrics) :
            score = performance(y_test, y_pred, metric)
            scores[m,k] = score
            
    return scores.mean(axis=1) # average across columns


def select_param_linear(X, y, kf, metrics=["accuracy"], plot=True) :
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting and metric,
    then selects the hyperparameter that maximizes the average performance for each metric.
    
    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf      -- model_selection.KFold or model_selection.StratifiedKFold
        metrics -- list of m strings, metrics
        plot    -- boolean, make a plot
    
    Returns
    --------------------
        params  -- list of m floats, optimal hyperparameter C for each metric
    """
    
    C_range = 10.0 ** np.arange(-3, 3)
    scores = np.empty((len(metrics), len(C_range)))

    # compute CV scores using cv_performance(...)
    for row, C in enumerate(C_range):
        print("C: ", C)
        scores[row] = cv_performance(SVC(C, kernel='linear'), X, y,
                             StratifiedKFold(n_splits=5), metrics)

    # get best hyperparameters
    best_params = [(0, 0) for i in metrics]    # dummy, okay to change
    for i, metric_scores in enumerate(scores):
        for j, score in enumerate(metric_scores):
            if score > best_params[j][0]:
                best_params[j] = score, C_range[i]
                
    # best_params = [tup[1] for tup in best_params]
    
    # plot
    if plot:
        plt.figure()
        ax = plt.gca()
        ax.set_ylim(0, 1)
        ax.set_xlabel("C")
        ax.set_ylabel("score")
        for m, metric in enumerate(metrics) :
            lineplot(C_range, scores[m,:], metric)
        plt.legend(loc='lower left')
        plt.show()
    
    return best_params


def select_param_rbf(X, y, kf, metrics=["accuracy"], plot=False) :
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
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
        params  -- list of m tuples, optimal hyperparameters (C,gamma) for each metric
    """
    
    # part 4b: for each metric, select optimal hyperparameters using cross-validation
    
    # create grid of hyperparameters
    # hint: use a small 2x2 grid of hyperparameters for debugging
    C_range = 10.0 ** np.arange(-3, 3)          # dummy, okay to change
    gamma_range = [0.25, 0.5, 0.75, 1, 1.5, 2, 5]    # dummy, okay to change
    #gamma_range = [0.1 * i for i in range(1,31)]
    scores = np.empty((len(metrics), len(C_range), len(gamma_range)))

    # compute CV scores using cv_performance(...)
    # get best hyperparameters
    for i, c_value in enumerate(C_range):
        for j, g_value in enumerate(gamma_range):
            print("C: ", c_value, "G: ", g_value)
            cv_perform =  cv_performance(SVC(c_value, kernel='rbf', gamma=g_value), X, y,
                                         StratifiedKFold(n_splits=5), metrics)
            for m, metric_score in enumerate(cv_perform):
                scores[m][i][j] = metric_score
        if plot:
            plt.figure()
            ax = plt.gca()
            ax.set_ylim(0, 1)
            ax.set_xlabel("G")
            ax.set_ylabel("score")

            for m, metric in enumerate(metrics):
                sc = []
                for g, gamma_v in enumerate(gamma_range):
                    sc.append(scores[m][i][g])
                lineplot(gamma_range, sc, metric)
            plt.suptitle("C_value of " + str(c_value), size=16)
            plt.legend(loc='lower left')
            plt.show()
        
    best_params = [(0,0,0) for i in metrics]
    for m, metric_scores in enumerate(scores):
        for c, c_values in enumerate(C_range):
            for g, g_values in enumerate(gamma_range):
                if metric_scores[c][g] > best_params[m][0]:
                    best_params[m] = metric_scores[c][g], c_values, g_values

    # best_params = [(tup[1], tup[2]) for tup in best_params]
    
    return best_params


def performance_CI(clf, X, y, metric="accuracy") :
    """
    Estimates the performance of the classifier using the 95% CI.
    
    Parameters
    --------------------
        clf          -- classifier (instance of SVC or DummyClassifier)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure
    
    Returns
    --------------------
        score        -- float, classifier performance
        lower        -- float, lower limit of confidence interval
        upper        -- float, upper limit of confidence interval
    """
    
    try :
        y_pred = clf.decision_function(X)
    except :  # for dummy classifiers
        y_pred = clf.predict(X)
    score = performance(y, y_pred, metric)
    
    ### ========== TODO : START ========== ###
    # part 5c: use bootstrapping to compute 95% confidence interval
    # hint: use np.random.randint(...) to sample

    n, d = X.shape
    t_performances = []
    t = 1000
    for i in range(t):
        indices = np.random.randint(0, n, size=n)
        X_sample = X[indices]
        y_sample = y[indices]
        sample_perfor = performance(y_sample, clf.predict(X), metric)
        t_performances.append(sample_perfor)
    t_performances.sort()
    score = 1.0 * sum(t_performances)/t
    low_ = (t_performances[int(t*0.025)] + t_performances[int(t*0.025)-1])/2
    high_ = (t_performances[int(t*0.975)] + t_performances[int(t*0.975)-1])/2
    return score, low_, high_
    ### ========== TODO : END ========== ###


######################################################################
# functions -- plotting
######################################################################

def lineplot(x, y, label):
    """
    Make a line plot.
    
    Parameters
    --------------------
        x            -- list of doubles, x values
        y            -- list of doubles, y values
        label        -- string, label for legend
    """
    
    xx = range(len(x))
    plt.plot(xx, y, linestyle='-', linewidth=2, label=label)
    plt.xticks(xx, x)


def plot_results(metrics, classifiers, *args):
    """
    Make a results plot.
    
    Parameters
    --------------------
        metrics      -- list of strings, metrics
        classifiers  -- list of strings, classifiers (excluding baseline classifier)
        args         -- variable length argument
                          results for baseline
                          results for classifier 1
                          results for classifier 2
                          ...
                        each results is a list of tuples ordered by metric
                          typically, each tuple consists of a single element, e.g. (score,)
                          to include error bars, each tuple consists of three elements, e.g. (score, lower, upper)
    """
    
    num_metrics = len(metrics)
    num_classifiers = len(args) - 1
    
    ind = np.arange(num_metrics)  # the x locations for the groups
    width = 0.7 / num_classifiers # the width of the bars
    
    fig, ax = plt.subplots()
    
    # loop through classifiers
    rects_list = []
    for i in xrange(num_classifiers):
        results = args[i+1] # skip baseline
        
        # mean
        means = [it[0] for it in results]
        rects = ax.bar(ind + i * width, means, width, label=classifiers[i])
        rects_list.append(rects)
        
        # errors
        if len(it) == 3:
            errs = [(it[0] - it[1], it[2] - it[0]) for it in results]
            ax.errorbar(ind + i * width, means, yerr=np.array(errs).T, fmt='none', ecolor='k')
    
    # baseline
    results = args[0]
    for i in xrange(num_metrics) :
        xlim = (ind[i] - 0.8 * width, ind[i] + num_classifiers * width - 0.2 * width)
        
        # mean
        mean = results[i][0]
        plt.plot(xlim, [mean, mean], color='k', linestyle='-', linewidth=2)
        
        # errors
        if len(results[i]) == 3:
            err_low = results[i][1]
            err_high = results[i][2]
            plt.plot(xlim, [err_low, err_low], color='k', linestyle='--', linewidth=2)
            plt.plot(xlim, [err_high, err_high], color='k', linestyle='--', linewidth=2)
    
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.set_xticks(ind + width / num_classifiers)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    def autolabel(rects):
        """Attach a text label above each bar displaying its height"""
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%.2f' % height, ha='center', va='bottom')
    
    for rects in rects_list:
        autolabel(rects)
    
    plt.show()
