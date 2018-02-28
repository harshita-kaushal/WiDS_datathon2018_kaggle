import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold, GridSearchCV
from  sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# vstack both train and test
train_test = pd.concat([train.drop(['is_female', 'train_id'], axis=1), test.drop(['test_id'], axis=1)], axis=0, ignore_index=True)  # (45540, 1233)

# remove features where at least one element is NaN (lazy approach and very strict criteria!)
# train_test2 = train_test.dropna(axis=1, how='any')  # (45540, 237)

# remove features where all elements are NaN 
# train_test2 = train_test.dropna(axis=1, how='all')  # (45540, 1217)

# remove features where at least 10000 elements are non-NaN 
train_test2 = train_test.dropna(axis=1, thresh=10000)  # (45540, 439)

# convert to integers (NaNs are converted to -1)
for i in train_test2:
    train_test2[i] = train_test2[i].astype("category").cat.codes  # (45540, 439)

# one hot encoding
train_test3 = pd.get_dummies(train_test2, columns=train_test2.columns)  # (45540, 5153)

# ====================================================================

X_train = train_test3.ix[:18254]
y_train = train[['is_female']]
X_test = train_test3.ix[18255:]

kf = KFold(n_splits=3, shuffle=False)

classifier = 'LogisticRegressionCV' # 'LogisticRegressionCV' or 'LinearSVC' or 'SVC' or 'SGDClassifier'

print('fitting')
# ====================================================================
# logistic regression
# ====================================================================
if classifier=='LogisticRegressionCV':  #results are identical when using sag or saga solvers, being slightly (1%) better than when using the default lbfgs
    clf = LogisticRegressionCV(cv=kf, n_jobs=-1, solver='lbfgs', max_iter=1000, verbose=0, penalty='l2')
    clf.fit(X_train, y_train)
    print(clf.scores_)
    print(clf.Cs_)
    print(clf.C_)


# ====================================================================
# SVC
# ====================================================================
if classifier=='SVC': #don't bother. way too slow.
    C_range = 10.**np.arange(-2, 2) #(-2, 6)  # (-2, 7) or (-2, 8) with more computing power
    gamma_range = 10.**np.arange(-9, -3) #(-9, -1)  # (-9, 0) or (-9, 1) ''
    param_dist = dict(C=C_range, gamma=gamma_range)
    grid_clf = GridSearchCV(SVC(cache_size=1000), param_dist, cv=kf, n_jobs=-1)
    clf.fit(X_train, y_train)
    print(clf.best_score_)


if classifier=='LinearSVC':
    C_range = 10.**np.arange(-2, 3) #(-2, 6)  # (-2, 7) or (-2, 8) with more computing power
    param_dist = dict(C=C_range)
    clf = GridSearchCV(LinearSVC(), param_dist, cv=kf, n_jobs=-1)
    clf.fit(X_train, y_train)
    print(clf.best_score_)

# ====================================================================
# SGDClassifier
# ====================================================================    
if classifier=='SGDClassifier':
    alpha_range = 10.**np.arange(-5, 2) #(-2, 6)  # (-2, 7) or (-2, 8) with more computing power
    param_dist = dict(alpha=alpha_range)
    clf = GridSearchCV(SGDClassifier(n_jobs=-1), param_dist, cv=kf)    
    clf.fit(X_train, y_train)
    print(clf.best_score_)
    
# ====================================================================
# RandomForests
# ====================================================================
if classifier=='RandomForestClassifier':
    n_estimators = np.array([1000]) # generally, the more the better
    max_depth = np.array([None]) # 1000
    param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)
    clf = GridSearchCV(RandomForestClassifier(n_jobs=-1, criterion='entropy'), param_grid, cv=3)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    print(clf.best_score_)
    
#cross_val_score(RandomForestClassifier(n_jobs=-1, criterion='entropy', n_estimators=1000, max_depth=1000), X_train, y_train, cv=3)

predictions = clf.predict(X_test)
result_table = pd.DataFrame(predictions, columns=['is_female'])
result_table.to_csv('submission.csv', columns=['is_female'], index_label='test_id')

print('done')










