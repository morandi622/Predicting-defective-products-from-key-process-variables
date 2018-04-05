
''''
REPLACE THE BELOW PATH TO YOUR DIRECTORY
'''
%cd "\\vmware-host\Shared Folders\Shared Folders\Desktop"



import numpy as np, matplotlib.pyplot as plt, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier

from matplotlib.colors import ListedColormap
cm=ListedColormap(['#0000aa', '#ff2020'])


df=pd.read_csv('xl data science.csv') #reading input file
y=(df.dropna()['target'].values>=1000).astype(int)  # target (1--> not defective, 0-->defective)
X=df.dropna().drop(['target'],axis=1).values  # features
pd.Series(y).value_counts() #counting not defective vs. defective items



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=42) #splitting dataset into train and test samples (stratified splitting)
ii,=np.where(y_test==0)

#routine to fit models and retrieve useful metrics (f-score(s), confusion matrix)
def stat(model,cross_val=False):
    if cross_val:
        grid = GridSearchCV(model, param_grid, cv=5, scoring='f1')
        grid.fit(X_train, y_train)
        classifier = grid.best_estimator_
    else:
        classifier=model.fit(X_train, y_train)

    pred = classifier.predict(X_test)
    confusion = confusion_matrix(y_test, pred)
    print("Confusion matrix:\n{}".format(confusion))

    print("Weighted average f1 score: {:.5f}".format(f1_score(y_test, pred)))
    print("Micro average f1 score: {:.5f}".format(f1_score(y_test, pred, average="micro")))
    print("Macro average f1 score: {:.5f}".format(f1_score(y_test, pred, average="macro")))

    print(classifier)




#plot feature importance or magnitude
def plot_feature(classifier):
    coef=classifier.feature_importances_
    feature_names = np.array(df.drop('target',axis=1).keys())
    colors = [cm(1) if c < 0 else cm(0) for c in coef]
    plt.ion()
    plt.figure(figsize=(15, 5))
    plt.bar(np.arange(len(coef)), coef, color=colors)
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(np.arange(len(coef)),feature_names, rotation=60,ha="right")
    plt.ylabel("Coefficient magnitude")
    plt.xlabel("Feature")
    plt.plot([-0.5,len(feature_names)-0.5],[0,0],'k--')
    plt.title(classifier.__class__.__name__)
    plt.tight_layout()


stat(DummyClassifier(strategy='most_frequent')) #DummyClassifier that always predict the majority class

pipe = make_pipeline(StandardScaler(), LinearSVC()) #standardize the data before fitting
param_grid = {'linearsvc__C': [10,25,50,80]}
stat(pipe, cross_val=True)


pipe = make_pipeline(StandardScaler(), LogisticRegression())
param_grid = {'logisticregression__C': [1, 10,50,80,100,150]}
stat(pipe, cross_val=True)


pipe = GradientBoostingClassifier(learning_rate=0.1,max_depth=3,max_features=2,n_estimators=300, random_state=42)
param_grid = {'max_depth': [2,3,4],'max_features': [2,3,4]}
stat(pipe, cross_val=True)


pipe=RandomForestClassifier(n_estimators=200,random_state=42)
param_grid = {'max_features': [3,4,5]}
stat(pipe, cross_val=True)



#Majorit voting classifier (hard voting). if ‘hard’, uses predicted class labels for majority rule voting. Else if ‘soft’, predicts the class label based on the argmax of the sums of the predicted probabilities
classifier= VotingClassifier([
    ('gbrt',GradientBoostingClassifier(learning_rate=0.1,max_depth=2,max_features=3,n_estimators=300, random_state=42)),
    ('rf',RandomForestClassifier(n_estimators=200,random_state=42,max_features=4))
], voting='hard')
stat(classifier)



#plotting  roc_curve
rf=RandomForestClassifier(n_estimators=200,random_state=42,max_features=4).fit(X_train, y_train)
from sklearn.metrics import roc_curve
plt.ion()
fpr, tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
# find threshold closest to zero
close_zero = np.argmin(np.abs(thresholds-0.5))
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
label="threshold 0.5", fillstyle="none", c='k', mew=2)
plt.legend(loc=4)
plt.title(rf.__class__.__name__)


#plot feature importance of RandomForest
plot_feature(rf)
