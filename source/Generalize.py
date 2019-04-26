
# coding: utf-8

# In[1]:


# Core imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from sklearn.multiclass import OneVsOneClassifier
# Preprocessing and visualization
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Metric functions
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import accuracy_score

# Models
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.dummy        import DummyClassifier
from sklearn.tree         import DecisionTreeClassifier

# Model selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Ignore warnings if they happen, we don't care (that much)
import warnings; warnings.simplefilter('ignore')

# Cross-validation takes a minute, so we will save these models
from sklearn.externals import joblib


# In[2]:


df     = pd.read_csv("data/lyrical_genius.csv")

# Remove pop songs, they are all over the place and hurt classification
df = df[(df["Genre"] != "pop")]

# Remove some irrelevant columns
df = df.drop(columns="Unnamed: 0")
df = df.drop(columns="Unnamed: 0.1")

# We go ahead and remove ALL duplicates
df = df.drop_duplicates(subset=["Name","Artist"],keep=False)

# Give each genre a new cool color
genres = df["Genre"].unique()
unique_colors = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'
]
colors = {}
i = 0
for genre in genres:
    colors[genre] = unique_colors[i]
    i+=1
    
# Upsample the amount of occurances of values that don't appear very often
# df = df.append(df[((df["Genre"] != "country") & (df["Genre"] != "edm_dance"))])
extras    = df.copy()
counts    = df["Genre"].value_counts()
max_count = max(df["Genre"].value_counts())
for genre in genres:
    needed = max_count - counts[genre]
    extras = extras.append(df[df["Genre"]==genre].sample(n=needed,replace=True))
df = extras
counts    = df["Genre"].value_counts()
colors_list = [colors[genre] for genre in genres]

df.head()


# In[3]:


# Split data into data frames of the right type
x_cols    = ["Is_Exp","Danceability","Energy","Key","Loudness","Mode","Speechiness","Acousticness","Instrumentalness","Liveness","Valence","Tempo","Time_Signature"]
y_cols    = ["Genre"]
meta_cols = ["Id","Popularity","Name","Artist"]

X,y,meta = df[x_cols],df[y_cols].iloc[:,0],df[meta_cols]


# In[4]:


# Scale the data and fit run 2D PCA on it
scaler   = StandardScaler()
scaled_X = scaler.fit_transform(X)
pca = PCA(n_components=2)
prin_comp = pca.fit_transform(scaled_X)
prin_df   = pd.DataFrame(data=prin_comp, columns=["PC1","PC2"])


# In[ ]:


# Split 
X_train, X_test, y_train, y_test = train_test_split(scaled_X,y, test_size=.2, random_state=1234, stratify=y)


# In[ ]:


clf = LogisticRegressionCV(cv=5, random_state=1234, multi_class="multinomial")
clf.fit(X_train,y_train)

# joblib.dump(clf, "saved_logistic.pkl")
y_pred = clf.predict(X_test)
training = clf.score(X_train, y_train)
testing  = clf.score(X_test,  y_test)
print("Training Accuracy: {}".format(training))
print("Testing  Accuracy: {}".format(testing))
print(genres)
print(confusion_matrix(y_test,y_pred,labels=genres))
print(classification_report(y_test, y_pred)) 


# In[ ]:


for i in range(len(clf.classes_)):
    print(clf.classes_[i])
    todos      = [(clf.coef_[i][j],list(X)[j]) for j in range(len(list(X)))]
    todos.sort(key=lambda x: abs(x[0]))
    print(todos[0])
    print(todos[1])
    
    print(todos[-2])
    print(todos[-1])
    


# In[ ]:


def best_classifier(X, y, t_clf, params) :
    """
    Sweeps different settings for the hyperparameters of a Decision Tree classifier,
    calculating the k-fold CV performance for each setting and metric,
    then selects the hyperparameters that maximize the average performance for each metric.
    """

    
    
    clf  = GridSearchCV(t_clf, params, cv=5,scoring= "accuracy")
    
    clf.fit(X,y)
    
    return clf.best_estimator_


# In[ ]:


weights = ["uniform","distance"]
pvals   = [1,2]
n_neigh = range(1,40,2)
params  = {
    "n_neighbors": range(1,40,2),
    "weights": weights,
    "p": pvals,

}
knn = KNeighborsClassifier()
clf = best_classifier(X_train,y_train,knn,params)

joblib.dump(clf, "saved_knn.pkl")

y_pred = clf.predict(X_test)
training = clf.score(X_train, y_train)
testing  = clf.score(X_test,  y_test)
print("Training Accuracy: {}".format(training))
print("Testing  Accuracy: {}".format(testing))
print(confusion_matrix(y_test,y_pred,labels=genres))
print(classification_report(y_test, y_pred)) 


# In[ ]:


params = {
    "max_depth": range(5,21),
    "min_samples_leaf": range(1,15),
    "criterion": ["gini","entropy"]
}
t = DecisionTreeClassifier()
print("traing dtre...")
DTree = best_classifier(X_train, y_train, t, params)

print("saving dtree...")
joblib.dump(DTree, "saved_dtree.pkl")
print("done.")
# predict genres of test data
accuracy = DTree.score(X_test,y_test)
y_pred = clf.predict(X_test)
training = DTree.score(X_train, y_train)
testing  = DTree.score(X_test,  y_test)
print("Training Accuracy: {}".format(training))
print("Testing  Accuracy: {}".format(testing))
print(confusion_matrix(y_test,y_pred,labels=genres))
print(classification_report(y_test, y_pred))


# In[ ]:


importance = DTree.feature_importances_
d_feats      = list(X)
todos      = [(importance[i],d_feats[i]) for i in range(len(d_feats))]
todos.sort(key=lambda x: x[0],reverse=True)


# In[ ]:


todos


# In[ ]:


# compare to stratified dummy classifier
dummy = DummyClassifier(strategy='stratified')

joblib.dump(dummy, "saved_dummy.pkl")

dummy.fit(X_train,y_train)
dummy_accuracy = dummy.score(X_test,y_test)
print( "Dummy classifier accuracy is" )
print(dummy_accuracy)

