"""
Author: Carlo Alberto Barbano <carlo.barbano@unito.it>
Date: 21/09/23
"""

import numpy as np
import multiprocessing
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, LeaveOneOut
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, balanced_accuracy_score, roc_auc_score

class AgeEstimator(BaseEstimator):
    """ Define the age estimator on latent space network features.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        print("AgeEstimator::reset: resetting estimator")
        self.age_estimator = self.init_estimator()

    def init_estimator(self):
        n_jobs = multiprocessing.cpu_count()
        return GridSearchCV(
            Ridge(random_state=0), param_grid={"alpha": 10.**np.arange(-2, 3)}, cv=5,
            scoring="r2", n_jobs=n_jobs)
        
    def fit(self, X, y):
        self.age_estimator.fit(X, y)
        return self.score(X, y)

    def cv(self, X, y, kfold = KFold(n_splits=10, shuffle=False)):
        scores = []
        y_true = []
        y_pred = []

        for train_index, test_index in kfold.split(X, y):
            self.reset()
            self.age_estimator.fit(X[train_index], y[train_index])
            scores.append(self.score(X[test_index], y[test_index]))
            y_true.extend(y[test_index])
            y_pred.extend(self.predict(X[test_index]))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        return np.mean(scores), np.std(scores), y_true, y_pred

    def predict(self, X):
        y_pred = self.age_estimator.predict(X)
        return y_pred
    
    def score(self, X, y):
        y_pred = self.age_estimator.predict(X)
        return mean_absolute_error(y, y_pred)
    
class AgeEstimatorAdjustedAverage(BaseEstimator):
    def __init__(self, age_estimator: AgeEstimator, train_estimator=False):
        super().__init__()
        self.age_estimator = age_estimator
        self.train_estimator = train_estimator
        self.reset()
    
    def reset(self):
        self.delta = 0.

    def cv(self, X, y, kfold=LeaveOneOut(), groups=None):
        scores = []
        y_true = []
        y_pred = []

        for train_idx, test_idx in kfold.split(X, y, groups=groups):
            self.reset()
            if self.train_estimator:
                self.age_estimator.reset()

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            self.fit(X_train, y_train)
            scores.append(self.score(X_test, y_test))
            y_true.extend(y_test)
            
            pred = self.predict(X_test, y_test)
            y_pred.extend(pred)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(scores), np.std(scores), y_true, y_pred

    def fit(self, X, y):
        if self.train_estimator:
            print("AgeEstimatorAdjustedAverage::fit: fitting base estimator")
            self.age_estimator.fit(X, y)
        
        y_pred = self.age_estimator.predict(X)
        self.delta = y.mean() - y_pred.mean()
        print("AgeEstimatorAdjustedAverage::fit: delta = %.3f" % self.delta)
        return self.score(X, y)
    
    def predict(self, X, y=None):
        return self.age_estimator.predict(X) + self.delta

    def score(self, X, y):
        y_pred = self.predict(X)
        return mean_absolute_error(y, y_pred)


class AgeEstimatorAdjustedDeLange(AgeEstimatorAdjustedAverage):
    def __init__(self, age_estimator: BaseEstimator, train_estimator=False):
        super().__init__(age_estimator, train_estimator=train_estimator)
    
    def reset(self):
        self.alpha, self.beta = 1., 0.

    def fit(self, X, y):
        if self.train_estimator:
            print(f"{self.__class__.__name__}::fit: fitting base estimator")
            self.age_estimator.fit(X, y)

        y_pred = self.age_estimator.predict(X)
        self.alpha, self.beta = np.linalg.lstsq(
            np.vstack([y, np.ones(len(y))]).T,
            y_pred,
            rcond=None
        )[0]
        print(f"{self.__class__.__name__}::fit: alpha = %.3f, beta = %.3f" % (self.alpha, self.beta))
        return self.score(X, y)
    def predict(self, X, y=None):
        return self.age_estimator.predict(X) + (y - (self.alpha * y + self.beta))

    def score(self, X, y=None):
        y_pred = self.predict(X, y)
        return mean_absolute_error(y, y_pred)


class AgeEstimatorAdjustedCole(AgeEstimatorAdjustedDeLange):
    def predict(self, X, y=None):
        return (self.age_estimator.predict(X) - self.beta) / self.alpha
    

class SiteEstimator(BaseEstimator):
    """ Define the site estimator on latent space network features.
    """
    def __init__(self):
        n_jobs = multiprocessing.cpu_count()
        self.site_estimator = GridSearchCV(
            LogisticRegression(solver="saga", max_iter=150), cv=5,
            param_grid={"C": 10.**np.arange(-2, 3)},
            scoring="balanced_accuracy", n_jobs=n_jobs)

    def fit(self, X, y):
        self.site_estimator.fit(X, y)
        return self.site_estimator.score(X, y)

    def predict(self, X):
        return self.site_estimator.predict(X)

    def score(self, X, y):
        return self.site_estimator.score(X, y)
    

class DiagnosisEstimator(BaseEstimator):
    def __init__(self):
        self.reset()

    def reset(self):
        self.estimator = self.init_estimator()

    def init_estimator(self):
        n_jobs = multiprocessing.cpu_count()
        return GridSearchCV(
            LogisticRegression(solver="saga", max_iter=10000, class_weight="balanced", random_state=0),
            cv=3, param_grid={"C": 10.**np.arange(-2, 3)},
            scoring="balanced_accuracy", n_jobs=n_jobs)

    def fit(self, X, y, scoring=accuracy_score):
        self.estimator.fit(X, y)
        return self.score(X, y, scoring=scoring)
    
    def cv(self, X, y, kfold=StratifiedKFold(n_splits=10, random_state=0, shuffle=True)):
        train_accuracy = []
        train_balanced_accuracy = []
        train_auc = []
        
        accuracy = []
        balanced_accuracy = []
        auc = []

        y_true = []
        y_pred = []

        for train_index, test_index in kfold.split(X, y):
            self.reset()
            self.estimator.fit(X[train_index], y[train_index])
            # y_pred_train = self.predict(y[train_index])
            # train_accuracy.append(accuracy_score(y[train_index], y_pred_train))
            
            pred = self.predict(X[test_index])
            accuracy.append(accuracy_score(y[test_index], pred))
            balanced_accuracy.append(balanced_accuracy_score(y[test_index], pred))

            y_proba = self.predict_proba(X[test_index])
            if y_proba.shape[1] == 2: # binary case
                y_proba = y_proba[:, 1]
            auc.append(roc_auc_score(y[test_index], y_proba, multi_class="ovo"))
            
            y_true.extend(y[test_index])
            y_pred.extend(pred)
        
        accuracy = np.array(accuracy)
        balanced_accuracy = np.array(balanced_accuracy)
        auc = np.array(auc)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        return accuracy.mean(), accuracy.std(), balanced_accuracy.mean(), balanced_accuracy.std(), \
               auc.mean(), auc.std(), y_true, y_pred
    
    def predict(self, X):
        y_pred = self.estimator.predict(X)
        return y_pred
    
    def predict_proba(self, X):
        y_proba = self.estimator.predict_proba(X)
        return y_proba
    
    def score(self, X, y, scoring=accuracy_score):
        y_pred = self.predict(X)
        return scoring(y, y_pred)
    
    def confusion_matrix(self, X, y):
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred)
    