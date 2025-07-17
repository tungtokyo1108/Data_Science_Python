#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 23:17:20 2022

    
"""

from abc import ABCMeta, abstractmethod
import numbers
import warnings

import numpy as np
import scipy.sparse as sp
from scipy import linalg
from scipy import optimize
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.special import expit 
from joblib import Parallel

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, MultiOutputMixin
from sklearn.preprocessing._data import _is_constant_feature
from sklearn.utils import check_array
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.extmath import _incremental_mean_and_var
from sklearn.utils.sparsefuncs import mean_variance_axis, inplace_column_scale
from sklearn.utils._seq_dataset import ArrayDataset32, CSRDataset32
from sklearn.utils._seq_dataset import ArrayDataset64, CSRDataset64
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.utils.fixes import delayed


SPARSE_INTERCEPT_DECAY = 0.01

def _deprecate_normalize(normalize, default, estimator_name):
    
    if normalize == "deprecated":
        _normalize = default
    else:
        _normalize = normalize
    
    alpha_msg = ""
    
    return _normalize

def make_dataset(X, y, sample_weight, random_state = None):
    
    """
    Create "Dataset" abstraction for sparse and dense inputs 
    """
    
    rng = check_random_state(random_state)
    seed = rng.randint(1, np.iinfo(np.int32).max)
    
    if X.dtype == np.float32:
        CSRData = CSRDataset32
        ArrayData = ArrayDataset32
    else:
        CSRData = CSRDataset64
        ArrayData = ArrayDataset64
        
    if sp.issparse(X):
        dataset = CSRData(X.data, X.indptr, X.indices, y, sample_weight, seed=seed)
        intercept_decay = SPARSE_INTERCEPT_DECAY
    else:
        X = np.ascontiguousarray(X)
        dataset = ArrayData(X, y, sample_weight, seed=seed)
        intercept_decay = 1.0
        
    return dataset, intercept_decay

def _preprocess_data(X, y, fit_intercept, normalize = False, copy = True, sample_weight = None, check_input = True):
    
    """
    Center and scale data
    Center data to have mean zero along axis 0. 
    """
    if isinstance(sample_weight, numbers.Number):
        sample_weight = None
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
    
    if check_input:
        X = check_array(X, copy=copy, accept_sparse=["csr", "csc"], dtype=FLOAT_DTYPES)
    elif copy:
        if sp.issparse(X):
            X = X.copy()
        else:
            X = X.copy(order = "K")
    
    y = np.asarray(y, dtype=X.dtype)
    
    if fit_intercept:
        if sp.issparse(X):
            X_offset, X_var = mean_variance_axis(X, axis = 0, weights = sample_weight)
        else:
            if normalize:
                X_offset, X_var, _ = _incremental_mean_and_var(X, last_mean = 0.0, last_variance = 0.0, 
                                        last_sample_count = 0.0, sample_weight = sample_weight)
            else:
                X_offset = np.average(X, axis = 0, weights = sample_weight)
            
            X_offset = X_offset.astype(X.dtype, copy = False)
            X -= X_offset
            
        if normalize:
            X_var = X_var.astype(X.dtype, copy = False)
            constant_mask = _is_constant_feature(X_var, X_offset, X.shape[0])
            if sample_weight is None:
                X_var *= X.shape[0]
            else:
                X_var *= sample_weight.sum()
            
            X_scale = np.sqrt(X_var, out = X_var)
            X_scale[constant_mask] = 1.0
            if sp.issparse(X):
                inplace_column_scale(X, 1.0 / X_scale)
            else:
                X /= X_scale
        else:
            X_scale = np.ones(X.shape[1], dtype = X.dtype)
        
        y_offset = np.average(y, axis = 0, weights = sample_weight)
        y = y - y_offset
    else: 
        X_offset = np.zeros(X.shape[1], dtype = X.dtype)
        X_scale = np.zeros(X.shape[1], dtype = X.dtype)
        if y.ndim == 1:
            y_offset = X.dtype.type(0)
        else:
            y_offset = np.zeros(y.shape[1], dtype = X.dtype)
    
    return X, y, X_offset, y_offset, X_scale

def _rescale_data(X, y, sample_weight):
    
    """
    Rescale data sample-wise by square root of sample_weight.
    
    For many linear models, this enables easy support for sample_weight because 
    
        (y - Xw)' S (y - Xw)
        
    with S = diag(sample_weight) becomes 
        
        || y_rescaled - x_rescaled w ||_2^2
    
    when setting
        
        y_rescaled = sqrt(S) y
        x_rescaled = sqrt(S) x
    
    """
    
    n_samples = X.shape[0]
    sample_weight = np.asarray(sample_weight)
    if sample_weight.ndim == 0:
        sample_weight = np.full(n_samples, sample_weight, dtype = sample_weight.dtype)
    sample_weight_sqrt = np.sqrt(sample_weight)
    sw_matrix = sparse.dia_matrix((sample_weight_sqrt, 0), shape=(n_samples, n_samples))
    X = safe_sparse_dot(sw_matrix, X)
    y = safe_sparse_dot(sw_matrix, y)
    
    return X, y, sample_weight_sqrt

class LinearModel(BaseEstimator, metaclass = ABCMeta):
    
        
    def _decision_function(self, X):
        
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset = False)
        return safe_sparse_dot(X, self.coef_.T, dense_output = True) + self.intercept_
    
    def predict(self, X):
        
        return self._decision_function(X)
    
    def _set_intercept(self, X_offset, y_offset, X_scale):
        
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_scale
            self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)
        else:
            self.intercept_ = 0.0
            
class LinearRegression(MultiOutputMixin, RegressorMixin, LinearModel):
    
    """
    Ordinary least squares Linear Regression
    
    Linear_Regression fits a linear model with coefficients w = (w1, ..., wp) 
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation. 
    """
    
    def __init__(
            self, 
            *, 
            fit_intercept=True, 
            normalize="deprecated", 
            copy_X=True, 
            n_jobs=None, 
            positive=False,
    ):
        
        """
        - fit_intercept: whether to calculate the intercept for this model. True/False, default = True.
        - normalize: regressors X will be normalized before regression by substracting the mean and dividing by the l2-norm. True/False, default = False.
        - copy_X: X will be copied.
        - n_jobs: parrallel computations.
        - positive: forces the cofficients to be positive.
        """
        
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
    
    def fit(self, X, y, sample_weight = None):
        
        """
        Step 1: check normalize input data
        """
        _normalize = _deprecate_normalize(self.normalize, default = False, estimator_name = self.__class__.__name__)
        
        # set up numbers of CPU cores for parallel computation
        n_jobs_ = self.n_jobs
        
        accept_sparse = False if self.positive else ["csr", "csc", "coo"]
        
        """
        Step 2: validate data 
        """
        X, y = self._validate_data(X, y, accept_sparse = accept_sparse, y_numeric = True, multi_output = True)
        
        """
        Step 3: All samples have similar weights
        """
        sample_weight = _check_sample_weight(sample_weight, X, dtype = X.dtype, only_non_negative = True)
        
        """
        Step 4: Preprocess input data
        """
        X, y, X_offset, y_offset, X_scale = _preprocess_data(X, y, 
                    fit_intercept=self.fit_intercept, normalize=_normalize, copy=self.copy_X, sample_weight=sample_weight)
        
        """
        Step 5: Rescale data
        """
        X, y, sample_weight_sqrt = _rescale_data(X, y, sample_weight)
        
        """
        Step 6: Use optimization algorithms to estimate parameters of model. There are 2 options:
            1 - non-negative least squares algorithm - nnls use for positive values for all parameters 
            2 - least-squares algorithm use for both postive and negative values of parameters 
        """
        if self.positive: 
            """
            We use non-negative least squares algorithm - nnls
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html#scipy.optimize.nnls 
            """
            if y.ndim < 2:
                self.coef_ = optimize.nnls(X, y)[0]
            else: 
                outs = Parallel(n_jobs = n_jobs_)(
                    delayed(optimize.nnls)(X, y[:,j]) for j in range(y.shape[1])
                    )
                self.coef_ = np.vstack([out[0] for out in outs])
        elif sp.issparse(X):
            """
            In case, X is a sparse matrix
            """
            X_offset_scale = X_offset / X_scale
            
            def matvec(b):
                return X.dot(b) - sample_weight_sqrt * b.dot()
            
            def rmatvec(b):
                return X.T.dot(b) - X_offset_scale * b.dot(sample_weight_sqrt)
            
            X_centered = sparse.linalg.LinearOperator(
                shape = X.shape, matvec = matvec, rmatvec = rmatvec)
            
            """
            The least-squares solution to a large, sparse, linear system of equations
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
            """
            if y.ndim < 2:
                self.coef_ = lsqr(X_centered, y)[0]
            else:
                outs = Parallel(n_jobs=n_jobs_)(
                    delayed(lsqr)(X_centered, y[:,j].ravel()) for j in range(y.shape[1]))
                self.coef_ = np.vstack([out[0] for out in outs])
        else:
            self.coef_, _, self.rank_, self.singular_ = linalg.lstsq(X, y)
            self.coef_ = self.coef_.T
        
        if y.ndim == 1:
            self.coef_ = np.ravel(self.coef_)
        self._set_intercept(X_offset, y_offset, X_scale)
        
        return self
                
        
##############################################################################################
## Test code
##############################################################################################

import pandas as pd
from sklearn.model_selection import train_test_split

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

reg = LinearRegression(n_jobs=2).fit(X, y)
reg.intercept_
reg.coef_
reg.predict(np.array([[3, 5]]))
      
#---------------------------------------------------------------------------------------------
  
bna = pd.read_csv("ROI_test.csv", index_col="BNAsubjID")
meta = pd.read_csv("Meta_test.csv", index_col="Subject")

# Regression 
y = meta["AgeTag"]    
    
X_train, X_test, y_train, y_test = train_test_split(bna, y, test_size=0.3, random_state=42)

reg = LinearRegression(n_jobs=2, positive=False).fit(X_train, y_train)    
reg.intercept_
reg.coef_  
reg.predict(X_test)  
    
