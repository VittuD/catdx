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
    """
    Predicts age based on latent features.
    Implements Ridge regression with hyperparameter tuning and cross-validation.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets the estimator by reinitializing the Ridge regression model.
        """
        print("AgeEstimator::reset: resetting estimator")
        self.age_estimator = self.init_estimator()

    def init_estimator(self):
        """
        Initializes a Ridge regression model with GridSearchCV for hyperparameter tuning.
        """
        n_jobs = multiprocessing.cpu_count()
        return GridSearchCV(
            Ridge(random_state=0), param_grid={"alpha": 10.**np.arange(-2, 3)}, cv=5,
            scoring="r2", n_jobs=n_jobs)

    def fit(self, X, y):
        """
        Fits the model to the training data and computes the mean absolute error.

        Parameters:
        X: array-like, feature matrix.
        y: array-like, target values.
        
        Returns:
        float: Mean Absolute Error (MAE).
        """
        self.age_estimator.fit(X, y)
        return self.score(X, y)

    def cv(self, X, y, kfold=KFold(n_splits=10, shuffle=False)):
        """
        Performs cross-validation and computes scores for each fold.

        Parameters:
        X: array-like, feature matrix.
        y: array-like, target values.
        kfold: KFold object, defines the cross-validation strategy.
        
        Returns:
        tuple: Mean and standard deviation of scores, true values, predicted values.
        """
        scores = []
        y_true = []
        y_pred = []

        for train_index, test_index in kfold.split(X, y):
            self.reset()
            self.age_estimator.fit(X[train_index], y[train_index])
            scores.append(self.score(X[test_index], y[test_index]))
            y_true.extend(y[test_index])
            y_pred.extend(self.predict(X[test_index]))

        return np.mean(scores), np.std(scores), np.array(y_true), np.array(y_pred)

    def predict(self, X):
        """
        Predicts target values using the trained model.

        Parameters:
        X: array-like, feature matrix.
        
        Returns:
        array: Predicted values.
        """
        return self.age_estimator.predict(X)

    def score(self, X, y):
        """
        Computes the mean absolute error between true and predicted values.

        Parameters:
        X: array-like, feature matrix.
        y: array-like, true target values.
        
        Returns:
        float: Mean Absolute Error (MAE).
        """
        y_pred = self.predict(X)
        return mean_absolute_error(y, y_pred)

class AgeEstimatorAdjustedAverage(BaseEstimator):
    """
    Adjusts predictions from AgeEstimator to account for systematic bias.
    """
    def __init__(self, age_estimator: AgeEstimator, train_estimator=False):
        super().__init__()
        self.age_estimator = age_estimator
        self.train_estimator = train_estimator
        self.reset()

    def reset(self):
        """
        Resets the adjustment term (delta) to zero.
        """
        self.delta = 0.

    def cv(self, X, y, kfold=LeaveOneOut(), groups=None):
        """
        Performs leave-one-out cross-validation with optional grouping.

        Parameters:
        X: array-like, feature matrix.
        y: array-like, target values.
        kfold: LeaveOneOut object, defines the cross-validation strategy.
        groups: Optional grouping information.
        
        Returns:
        tuple: Mean and standard deviation of scores, true values, predicted values.
        """
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

        return np.mean(scores), np.std(scores), np.array(y_true), np.array(y_pred)

    def fit(self, X, y):
        """
        Fits the base estimator and computes the adjustment term (delta).

        Parameters:
        X: array-like, feature matrix.
        y: array-like, target values.
        
        Returns:
        float: Mean Absolute Error (MAE).
        """
        if self.train_estimator:
            print("AgeEstimatorAdjustedAverage::fit: fitting base estimator")
            self.age_estimator.fit(X, y)

        y_pred = self.age_estimator.predict(X)
        self.delta = y.mean() - y_pred.mean()
        print(f"AgeEstimatorAdjustedAverage::fit: delta = {self.delta:.3f}")
        return self.score(X, y)

    def predict(self, X, y=None):
        """
        Adjusts predictions using the computed delta.

        Parameters:
        X: array-like, feature matrix.
        y: Optional, true target values.
        
        Returns:
        array: Adjusted predictions.
        """
        return self.age_estimator.predict(X) + self.delta

    def score(self, X, y):
        """
        Computes the mean absolute error for adjusted predictions.

        Parameters:
        X: array-like, feature matrix.
        y: array-like, true target values.
        
        Returns:
        float: Mean Absolute Error (MAE).
        """
        y_pred = self.predict(X)
        return mean_absolute_error(y, y_pred)
