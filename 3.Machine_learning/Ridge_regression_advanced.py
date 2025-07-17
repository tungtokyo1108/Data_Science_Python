#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 22:56:53 2022

@author: tungdang
"""

from abc import ABCMeta, abstractmethod
from functools import partial
import warnings

import numpy as np
import numbers
from scipy import linalg
from scipy import sparse
from scipy import optimize
from scipy.sparse import linalg as sp_linalg

from sklearn.linear_model._base import LinearClassifierMixin, LinearModel
from sklearn.linear_model._base import _deprecate_normalize, _preprocess_data, _rescale_data
from sklearn.linear_model._sag import sag_solver
from sklearn.base import MultiOutputMixin, RegressorMixin, is_classifier
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.extmath import row_norms
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils import check_scalar
from sklearn.utils import compute_sample_weight
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _check_sample_weight
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import check_scoring
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.sparsefuncs import mean_variance_axis


def _get_rescaled_operator(X, X_offset, sample_weight_sqrt):
    
    def matvec(b):
        return X.dot(b) - sample_weight_sqrt * b.dot(X_offset)
    
    def rmatvec(b):
        return X.T.dot(b) - X_offset * b.dot(sample_weight_sqrt)
    
    X1 = sparse.linalg.LinearOperator(shape=X.shape, matvec = matvec, rmatvec = rmatvec)
    
    return X1

def _solve_sparse_cg(X, y, alpha, max_iter=None, tol=1e-3, verbose=0, X_offset=None, X_scale=None, sample_weight_sqrt=None):
    
    if sample_weight_sqrt is None:
        sample_weight_sqrt = np.ones(X.shape[0], dtype=X.dtype)
    
    n_samples, n_features = X.shape
    
    if X_offset is None or X_scale is None:
        X1 = sp_linalg.aslinearoperator(X)
    else:
        X_offset_scale = X_offset / X_scale 
        X1 = _get_rescaled_operator(X, X_offset_scale, sample_weight_sqrt)
    
    coefs = np.empty((y.shape[1], n_features), dtype=X.dtype)
    
    if n_features > n_samples:
        
        def create_mv(curr_alpha):
            def _mv(x):
                return X1.matvec(X1.rmatvec(x)) + curr_alpha * x
            
            return _mv 
    else:
        def create_mv(curr_alpha):
            def _mv(x):
                return X1.rmatmat(X1.matvec(x)) + curr_alpha * x
            
            return _mv 
    
    for i in range(y.shape[1]):
        y_column = y[:,i]
        
        mv = create_mv(alpha[i])
        if n_features > n_samples:
            # kernel ridge
            # w = X.T * inv(X X^t + alpha*Id) y 
            C = sp_linalg.LinearOperator((n_samples, n_samples), matvec=mv, dtype=X.dtype)
            
            try:
                coef, info = sp_linalg.cg(C, y_column, tol=tol, atol="legacy")
            except TypeError:
                coef, info = sp_linalg.cg(C, y_column, tol=tol)
                
            coefs[i] = X1.rmatvec(coef)
            
        else: 
            # Linear ridge
            # w = inv(X^t X + alpha * Id) * X.T y
            y_column = X1.rmatvec(y_column)
            C = sp_linalg.LinearOperator((n_features, n_features), matvec=mv, dtype=X.dtype)
            
            try:
                coefs[i], info = sp_linalg.cg(C, y_column, maxiter=max_iter, tol=tol, atol="legacy")
            except:
                coefs[i], info = sp_linalg.cg(C, y_column, maxiter=max_iter, tol=tol)
    
    return coefs
        













































