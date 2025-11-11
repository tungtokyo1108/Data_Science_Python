#!/usr/bin/env python3
"""
Machine Learning Models for Nikkei Index Prediction
Comprehensive implementation of state-of-the-art ML models including:
- XGBoost
- LightGBM
- Random Forest
- Support Vector Regression (SVR)
- Gaussian Process Regression (GPR)
- Ensemble Method

Following the exact format of LSTM comparison framework with candlestick visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
import pickle
import os
from datetime import datetime
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.visualization import plot_contour, plot_slice
import shap

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import data handling from the existing framework
from test_nikkei_autocon_clean import (
    NikkeiDataDownloader, PerformanceTester
)


class MLModelsComparison:
    """
    Comprehensive ML models comparison framework for Nikkei index prediction
    Following the exact structure of LSTMAutoConComparison
    """

    def __init__(self, prediction_mode='return'):
        """
        Initialize ML Models Comparison Framework

        Args:
            prediction_mode: Type of prediction
                - 'return': Predict average return (percentage change) - DEFAULT
                - 'price': Predict direct close price values
        """
        self.downloader = NikkeiDataDownloader()
        self.tester = PerformanceTester()
        self.results = {}
        self.trained_models = {}
        self.scalers = {}
        self.best_params = {}  # Store best hyperparameters from optimization
        self.optimization_studies = {}  # Store Optuna study objects

        # Prediction mode configuration
        if prediction_mode not in ['return', 'price']:
            raise ValueError("prediction_mode must be 'return' or 'price'")
        self.prediction_mode = prediction_mode

        print(f"\n{'='*70}")
        print(f"ML Models Comparison Framework Initialized")
        print(f"Prediction Mode: {prediction_mode.upper()}")
        if prediction_mode == 'return':
            print(f"  → Target: Average return over prediction window")
            print(f"  → Output: Percentage change (e.g., 0.02 = 2% gain)")
            print(f"  → Features: All EXCEPT 'close' price (prevents data leakage)")
        else:
            print(f"  → Target: NEXT day's close price")
            print(f"  → Output: Direct price value (¥)")
            print(f"  → Features: All EXCEPT 'close' price (prevents data leakage)")
        print(f"{'='*70}\n")

    def prepare_data(self, start_date="2020-01-01", end_date=None):
        """Prepare data for ML models using the same methodology as LSTM"""
        print("📊 Preparing Data for ML Models Comparison...")

        # Use the same data preparation methodology
        processed_data = self.tester.prepare_data(start_date, end_date)

        # Store data splits
        self.train_data = self.tester.train_data
        self.val_data = self.tester.val_data
        self.test_data = self.tester.test_data

        print(f"📊 Data prepared for ML models:")
        print(f"   Training: {len(self.train_data)} samples")
        print(f"   Validation: {len(self.val_data)} samples")
        print(f"   Testing: {len(self.test_data)} samples")

        return processed_data

    def create_sequences(self, data, input_length=60, output_length=5):
        """
        Create sequences for ML models (similar to LSTM dataset creation)

        IMPORTANT: Excludes 'close' price from input features to prevent data leakage!
        - Input features: All columns EXCEPT 'close' and 'date'
        - Target: 'close' price (future values)

        Supports two prediction modes:
        - 'return': Predict average return (percentage change)
        - 'price': Predict direct close price

        Returns:
            X: Input sequences (flattened features WITHOUT close price)
            y: Target values (returns or prices depending on mode)
        """
        X, y = [], []

        # CRITICAL: Exclude 'close' from input features to prevent data leakage
        # Only use other features (open, high, low, volume, indicators, etc.)
        feature_cols = [col for col in data.columns if col not in ['date', 'close']]

        print(f"   Input features (excluding close): {len(feature_cols)} columns")
        print(f"   Features: {', '.join(feature_cols[:5])}{'...' if len(feature_cols) > 5 else ''}")

        for i in range(len(data) - input_length - output_length + 1):
            # Input sequence: flatten the windowed features (WITHOUT close price)
            # Uses data from time [i] to [i+input_length-1]
            X_seq = data[feature_cols].iloc[i:i+input_length].values.flatten()

            # Define target based on prediction mode
            if self.prediction_mode == 'return':
                # Mode 1: Predict RETURN (percentage change)
                # Current price: last price in input window
                current_price = data['close'].iloc[i+input_length-1]
                # Future prices: next output_length days after input window
                future_prices = data['close'].iloc[i+input_length:i+input_length+output_length].values
                # Target: average return over prediction window
                avg_return = np.mean((future_prices - current_price) / current_price)
                target = avg_return

            else:  # prediction_mode == 'price'
                # Mode 2: Predict DIRECT PRICE
                # Target: the NEXT close price immediately after input window
                # This is at time [i+input_length] (first day after input window)
                next_close_price = data['close'].iloc[i+input_length]
                target = next_close_price

            X.append(X_seq)
            y.append(target)

        return np.array(X), np.array(y)

    def optimize_xgboost_hyperparameters(self, input_length=60, output_length=5, n_trials=100):
        """
        Optimize XGBoost hyperparameters using Bayesian Optimization (Optuna)

        Args:
            input_length: Number of historical days for input
            output_length: Number of days to predict
            n_trials: Number of optimization trials (default: 100)

        Returns:
            best_params: Dictionary of optimized hyperparameters
        """
        print("\n🔍 Optimizing XGBoost Hyperparameters with Bayesian Optimization...")
        print("=" * 70)

        # Create sequences
        X_train, y_train = self.create_sequences(self.train_data, input_length, output_length)
        X_val, y_val = self.create_sequences(self.val_data, input_length, output_length)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Number of trials: {n_trials}")
        print(f"   Optimization metric: Validation MAE")

        def objective(trial):
            """Optuna objective function for XGBoost"""
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'objective': 'reg:squarederror',
                'random_state': 42,
                'n_jobs': -1
            }

            model = xgb.XGBRegressor(**params)
            model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)

            # Predict and calculate validation MAE
            val_pred = model.predict(X_val_scaled)
            val_mae = mean_absolute_error(y_val, val_pred)

            # Print progress every trial
            print(f"      Trial {trial.number + 1}/{n_trials} - MAE: {val_mae:.6f} - Params: n_est={params['n_estimators']}, depth={params['max_depth']}, lr={params['learning_rate']:.4f}")

            return val_mae

        # Create Optuna study with callback for better progress tracking
        study = optuna.create_study(
            direction='minimize',
            study_name='xgboost_optimization',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Optimize with custom callback
        print(f"\n   Starting optimization... (this may take 15-30 minutes)")
        print(f"   Each trial tests a different hyperparameter combination")
        print(f"   Progress will be printed for each trial:\n")

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Store results
        self.optimization_studies['XGBoost'] = study
        self.best_params['XGBoost'] = study.best_params

        print(f"\n✅ Optimization Complete!")
        print(f"   Best Validation MAE: {study.best_value:.6f}")
        print(f"   Best Hyperparameters:")
        for param, value in study.best_params.items():
            print(f"      {param}: {value}")

        return study.best_params

    def optimize_lightgbm_hyperparameters(self, input_length=60, output_length=5, n_trials=100):
        """
        Optimize LightGBM hyperparameters using Bayesian Optimization (Optuna)

        Args:
            input_length: Number of historical days for input
            output_length: Number of days to predict
            n_trials: Number of optimization trials (default: 100)

        Returns:
            best_params: Dictionary of optimized hyperparameters
        """
        print("\n🔍 Optimizing LightGBM Hyperparameters with Bayesian Optimization...")
        print("=" * 70)

        # Create sequences
        X_train, y_train = self.create_sequences(self.train_data, input_length, output_length)
        X_val, y_val = self.create_sequences(self.val_data, input_length, output_length)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Number of trials: {n_trials}")
        print(f"   Optimization metric: Validation MAE")

        def objective(trial):
            """Optuna objective function for LightGBM"""
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True),
                'objective': 'regression',
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )

            # Predict and calculate validation MAE
            val_pred = model.predict(X_val_scaled)
            val_mae = mean_absolute_error(y_val, val_pred)

            # Print progress every trial
            print(f"      Trial {trial.number + 1}/{n_trials} - MAE: {val_mae:.6f} - Params: n_est={params['n_estimators']}, leaves={params['num_leaves']}, lr={params['learning_rate']:.4f}")

            return val_mae

        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            study_name='lightgbm_optimization',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Optimize
        print(f"\n   Starting optimization... (this may take 15-30 minutes)")
        print(f"   Each trial tests a different hyperparameter combination")
        print(f"   Progress will be printed for each trial:\n")

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Store results
        self.optimization_studies['LightGBM'] = study
        self.best_params['LightGBM'] = study.best_params

        print(f"\n✅ Optimization Complete!")
        print(f"   Best Validation MAE: {study.best_value:.6f}")
        print(f"   Best Hyperparameters:")
        for param, value in study.best_params.items():
            print(f"      {param}: {value}")

        return study.best_params

    def optimize_randomforest_hyperparameters(self, input_length=60, output_length=5, n_trials=100):
        """
        Optimize Random Forest hyperparameters using Bayesian Optimization (Optuna)

        Args:
            input_length: Number of historical days for input
            output_length: Number of days to predict
            n_trials: Number of optimization trials (default: 100)

        Returns:
            best_params: Dictionary of optimized hyperparameters
        """
        print("\n🔍 Optimizing Random Forest Hyperparameters with Bayesian Optimization...")
        print("=" * 70)

        # Create sequences
        X_train, y_train = self.create_sequences(self.train_data, input_length, output_length)
        X_val, y_val = self.create_sequences(self.val_data, input_length, output_length)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Number of trials: {n_trials}")
        print(f"   Optimization metric: Validation MAE")

        def objective(trial):
            """Optuna objective function for Random Forest"""
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.1),
                'random_state': 42,
                'n_jobs': -1
            }

            model = RandomForestRegressor(**params)
            model.fit(X_train_scaled, y_train)

            # Predict and calculate validation MAE
            val_pred = model.predict(X_val_scaled)
            val_mae = mean_absolute_error(y_val, val_pred)

            # Print progress every trial
            print(f"      Trial {trial.number + 1}/{n_trials} - MAE: {val_mae:.6f} - Params: n_est={params['n_estimators']}, depth={params['max_depth']}, features={params['max_features']}")

            return val_mae

        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            study_name='randomforest_optimization',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Optimize
        print(f"\n   Starting optimization... (this may take 20-40 minutes)")
        print(f"   Each trial tests a different hyperparameter combination")
        print(f"   Progress will be printed for each trial:\n")

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Store results
        self.optimization_studies['RandomForest'] = study
        self.best_params['RandomForest'] = study.best_params

        print(f"\n✅ Optimization Complete!")
        print(f"   Best Validation MAE: {study.best_value:.6f}")
        print(f"   Best Hyperparameters:")
        for param, value in study.best_params.items():
            print(f"      {param}: {value}")

        return study.best_params

    def optimize_svr_hyperparameters(self, input_length=60, output_length=5, n_trials=50):
        """
        Optimize SVR hyperparameters using Bayesian Optimization (Optuna)
        Note: SVR is computationally expensive, so fewer trials recommended

        Args:
            input_length: Number of historical days for input
            output_length: Number of days to predict
            n_trials: Number of optimization trials (default: 50, less than other models)

        Returns:
            best_params: Dictionary of optimized hyperparameters
        """
        print("\n🔍 Optimizing SVR Hyperparameters with Bayesian Optimization...")
        print("=" * 70)

        # Create sequences
        X_train, y_train = self.create_sequences(self.train_data, input_length, output_length)
        X_val, y_val = self.create_sequences(self.val_data, input_length, output_length)

        # Scale features (critical for SVR)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Number of trials: {n_trials}")
        print(f"   Optimization metric: Validation MAE")
        print(f"   Note: SVR optimization is slower due to computational complexity")

        def objective(trial):
            """Optuna objective function for SVR"""
            kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])

            params = {
                'kernel': kernel,
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'epsilon': trial.suggest_float('epsilon', 1e-4, 1.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'cache_size': 1000
            }

            # Add degree parameter only for polynomial kernel
            if kernel == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 5)

            model = SVR(**params)
            model.fit(X_train_scaled, y_train)

            # Predict and calculate validation MAE
            val_pred = model.predict(X_val_scaled)
            val_mae = mean_absolute_error(y_val, val_pred)

            # Print progress every trial
            print(f"      Trial {trial.number + 1}/{n_trials} - MAE: {val_mae:.6f} - Params: kernel={kernel}, C={params['C']:.4f}, epsilon={params['epsilon']:.4f}")

            return val_mae

        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            study_name='svr_optimization',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Optimize
        print(f"\n   Starting optimization... (this may take 30-60 minutes, SVR is slow)")
        print(f"   Each trial tests a different hyperparameter combination")
        print(f"   Progress will be printed for each trial:\n")

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Store results
        self.optimization_studies['SVR'] = study
        self.best_params['SVR'] = study.best_params

        print(f"\n✅ Optimization Complete!")
        print(f"   Best Validation MAE: {study.best_value:.6f}")
        print(f"   Best Hyperparameters:")
        for param, value in study.best_params.items():
            print(f"      {param}: {value}")

        return study.best_params

    def optimize_gaussian_process_hyperparameters(self, input_length=60, output_length=5, n_trials=50, max_samples=1000):
        """
        Optimize Gaussian Process Regression hyperparameters using Bayesian Optimization (Optuna)

        Note: GPR is computationally expensive, so we use:
        - Fewer trials (default: 50)
        - Subset of training data (default: 1000 samples)
        - Optimized kernel selection

        Args:
            input_length: Number of historical days for input
            output_length: Number of days to predict
            n_trials: Number of optimization trials (default: 50, less than other models)
            max_samples: Maximum training samples to use (default: 1000)

        Returns:
            best_params: Dictionary of optimized hyperparameters including kernel configuration
        """
        print("\n🔍 Optimizing Gaussian Process Regression Hyperparameters with Bayesian Optimization...")
        print("=" * 70)

        # Create sequences
        X_train, y_train = self.create_sequences(self.train_data, input_length, output_length)
        X_val, y_val = self.create_sequences(self.val_data, input_length, output_length)

        # For GPR, use a subset of training data due to computational complexity O(n^3)
        # Sample strategically: more recent data + random samples
        if len(X_train) > max_samples:
            # Take last half of max_samples (most recent) + random samples from earlier
            recent_size = max_samples // 2
            early_size = max_samples - recent_size
            recent_indices = np.arange(len(X_train) - recent_size, len(X_train))
            early_indices = np.random.choice(len(X_train) - recent_size, size=early_size, replace=False)
            sample_indices = np.concatenate([early_indices, recent_indices])
            X_train_sampled = X_train[sample_indices]
            y_train_sampled = y_train[sample_indices]
        else:
            X_train_sampled = X_train
            y_train_sampled = y_train

        # Scale features (critical for GPR)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sampled)
        X_val_scaled = scaler.transform(X_val)

        print(f"   Training samples: {len(X_train_scaled)} (sampled from {len(X_train)})")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Feature dimension: {X_train.shape[1]}")
        print(f"   Number of trials: {n_trials}")
        print(f"   Optimization metric: Validation MAE")
        print(f"   Note: GPR optimization is slow due to O(n^3) complexity")

        def objective(trial):
            """Optuna objective function for Gaussian Process Regression"""

            # 1. Choose kernel type
            kernel_type = trial.suggest_categorical('kernel_type', ['RBF', 'Matern', 'RationalQuadratic', 'RBF+Matern'])

            # 2. ConstantKernel parameters (amplitude)
            constant_value = trial.suggest_float('constant_value', 0.1, 10.0)
            constant_bounds_low = trial.suggest_float('constant_bounds_low', 1e-3, 0.1, log=True)
            constant_bounds_high = trial.suggest_float('constant_bounds_high', 10.0, 1e3, log=True)

            # 3. Length scale parameters (for RBF and Matern kernels)
            length_scale = trial.suggest_float('length_scale', 0.1, 10.0)
            length_scale_bounds_low = trial.suggest_float('length_scale_bounds_low', 1e-2, 0.1, log=True)
            length_scale_bounds_high = trial.suggest_float('length_scale_bounds_high', 10.0, 1e2, log=True)

            # 4. WhiteKernel (noise) parameters
            noise_level = trial.suggest_float('noise_level', 1e-7, 1e-3, log=True)
            noise_bounds_low = trial.suggest_float('noise_bounds_low', 1e-10, 1e-5, log=True)
            noise_bounds_high = trial.suggest_float('noise_bounds_high', 1e-3, 1e-1, log=True)

            # 5. Model parameters
            alpha = trial.suggest_float('alpha', 1e-10, 1e-4, log=True)
            n_restarts_optimizer = trial.suggest_int('n_restarts_optimizer', 1, 10)
            normalize_y = trial.suggest_categorical('normalize_y', [True, False])

            # Build kernel based on selected type
            constant_kernel = ConstantKernel(
                constant_value=constant_value,
                constant_value_bounds=(constant_bounds_low, constant_bounds_high)
            )

            if kernel_type == 'RBF':
                main_kernel = RBF(
                    length_scale=length_scale,
                    length_scale_bounds=(length_scale_bounds_low, length_scale_bounds_high)
                )
            elif kernel_type == 'Matern':
                # Matern kernel with nu parameter
                nu = trial.suggest_categorical('matern_nu', [0.5, 1.5, 2.5])
                main_kernel = Matern(
                    length_scale=length_scale,
                    length_scale_bounds=(length_scale_bounds_low, length_scale_bounds_high),
                    nu=nu
                )
            elif kernel_type == 'RationalQuadratic':
                # RationalQuadratic kernel
                from sklearn.gaussian_process.kernels import RationalQuadratic
                alpha_rq = trial.suggest_float('alpha_rq', 0.1, 10.0)
                main_kernel = RationalQuadratic(
                    length_scale=length_scale,
                    alpha=alpha_rq,
                    length_scale_bounds=(length_scale_bounds_low, length_scale_bounds_high)
                )
            else:  # RBF+Matern combination
                nu = trial.suggest_categorical('matern_nu', [0.5, 1.5, 2.5])
                rbf_kernel = RBF(
                    length_scale=length_scale,
                    length_scale_bounds=(length_scale_bounds_low, length_scale_bounds_high)
                )
                matern_kernel = Matern(
                    length_scale=length_scale * 0.5,
                    length_scale_bounds=(length_scale_bounds_low, length_scale_bounds_high),
                    nu=nu
                )
                main_kernel = rbf_kernel + matern_kernel

            # Add white noise kernel
            white_kernel = WhiteKernel(
                noise_level=noise_level,
                noise_level_bounds=(noise_bounds_low, noise_bounds_high)
            )

            # Complete kernel: Constant * Main + White (noise)
            kernel = constant_kernel * main_kernel + white_kernel

            # Create and train model
            try:
                model = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=alpha,
                    n_restarts_optimizer=n_restarts_optimizer,
                    normalize_y=normalize_y,
                    random_state=42
                )

                model.fit(X_train_scaled, y_train_sampled)

                # Predict and calculate validation MAE
                val_pred = model.predict(X_val_scaled)
                val_mae = mean_absolute_error(y_val, val_pred)

                # Print progress every trial
                print(f"      Trial {trial.number + 1}/{n_trials} - MAE: {val_mae:.6f} - Kernel: {kernel_type}, alpha: {alpha:.2e}, normalize: {normalize_y}")

                return val_mae

            except Exception as e:
                # If training fails (e.g., numerical issues), return a large penalty
                print(f"      Trial {trial.number + 1}/{n_trials} - FAILED: {str(e)[:50]}")
                return float('inf')

        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            study_name='gpr_optimization',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Optimize
        print(f"\n   Starting optimization... (this may take 30-60 minutes, GPR is very slow)")
        print(f"   Each trial tests a different kernel and hyperparameter combination")
        print(f"   Progress will be printed for each trial:\n")

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Store results
        self.optimization_studies['GPR'] = study
        self.best_params['GPR'] = study.best_params

        print(f"\n✅ Optimization Complete!")
        print(f"   Best Validation MAE: {study.best_value:.6f}")
        print(f"   Best Hyperparameters:")
        for param, value in study.best_params.items():
            if isinstance(value, float):
                print(f"      {param}: {value:.6e}")
            else:
                print(f"      {param}: {value}")

        return study.best_params

    def explain_model_with_shap(self, model_name, num_samples=500, save_plots=True):
        """
        Generate comprehensive SHAP (SHapley Additive exPlanations) interpretability analysis

        SHAP provides both local (individual prediction) and global (overall model) explanations
        using game theory to fairly attribute prediction contributions to each feature.

        Args:
            model_name: Name of the model to explain ('XGBoost', 'LightGBM', 'RandomForest', etc.)
            num_samples: Number of samples to use for SHAP analysis (default: 500)
            save_plots: Whether to save plots to disk (default: True)

        Returns:
            dict: Dictionary containing SHAP values and feature information
        """
        print(f"\n🔍 Generating SHAP Interpretability Analysis for {model_name}...")
        print("=" * 70)

        if model_name not in self.trained_models:
            print(f"❌ Model {model_name} not found. Please train the model first.")
            return None

        model = self.trained_models[model_name]
        scaler = self.scalers.get(model_name)

        if scaler is None and model_name != 'Ensemble':
            print(f"❌ Scaler not found for {model_name}")
            return None

        # Use test data for SHAP analysis
        X_test, y_test = self.create_sequences(self.test_data, input_length=60, output_length=5)
        X_test_scaled = scaler.transform(X_test)

        # Sample data if too large (SHAP can be slow)
        if len(X_test_scaled) > num_samples:
            sample_indices = np.random.choice(len(X_test_scaled), size=num_samples, replace=False)
            X_sample = X_test_scaled[sample_indices]
            y_sample = y_test[sample_indices]
        else:
            X_sample = X_test_scaled
            y_sample = y_test

        print(f"   Analyzing {len(X_sample)} samples from test set")
        print(f"   Feature dimension: {X_sample.shape[1]}")

        # Generate feature names (flattened time series features)
        # CRITICAL: Exclude 'close' from features (same as training)
        feature_cols = [col for col in self.test_data.columns if col not in ['date', 'close']]
        feature_names = []
        for t in range(60):  # input_length = 60
            for col in feature_cols:
                feature_names.append(f"{col}_t-{60-t}")

        print(f"   ⚠️  NOTE: 'close' price excluded from input features (preventing data leakage)")
        print(f"   Input features: {len(feature_cols)} columns × 60 time steps = {len(feature_names)} total features")

        # Create DataFrame for better visualization
        X_sample_df = pd.DataFrame(X_sample, columns=feature_names)

        print(f"   Creating SHAP explainer...")

        # Use TreeExplainer for tree-based models (much faster)
        if model_name in ['XGBoost', 'LightGBM', 'RandomForest']:
            explainer = shap.TreeExplainer(model)
            print(f"   Using TreeExplainer (optimized for {model_name})")
        elif model_name == 'SVR':
            # Use KernelExplainer for SVR (slower but model-agnostic)
            background = shap.sample(X_sample_df, min(100, len(X_sample_df)))
            explainer = shap.KernelExplainer(model.predict, background)
            print(f"   Using KernelExplainer (model-agnostic for SVR)")
        elif model_name == 'GPR':
            # Use KernelExplainer for GPR
            background = shap.sample(X_sample_df, min(100, len(X_sample_df)))
            explainer = shap.KernelExplainer(lambda x: model.predict(x)[0] if hasattr(model.predict(x), '__len__') else model.predict(x), background)
            print(f"   Using KernelExplainer (model-agnostic for GPR)")
        else:
            print(f"   ⚠️ Model type not supported for SHAP analysis")
            return None

        print(f"   Computing SHAP values... (this may take a few minutes)")
        shap_values = explainer.shap_values(X_sample_df)

        # If shap_values is a list (multi-class), take the first element
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        print(f"   ✅ SHAP values computed successfully!")

        # Create comprehensive visualizations
        self._create_shap_visualizations(
            shap_values, X_sample_df, feature_names, model_name, save_plots
        )

        # Calculate and print feature importance
        feature_importance = self._calculate_shap_feature_importance(
            shap_values, feature_names
        )

        print(f"\n📊 Top 10 Most Important Features (by mean |SHAP value|):")
        for i, (feat, importance) in enumerate(feature_importance[:10], 1):
            print(f"   {i}. {feat}: {importance:.6f}")

        return {
            'shap_values': shap_values,
            'feature_names': feature_names,
            'feature_importance': feature_importance,
            'explainer': explainer,
            'X_sample': X_sample_df
        }

    def _create_shap_visualizations(self, shap_values, X_sample_df, feature_names, model_name, save_plots):
        """Create comprehensive SHAP visualization plots"""

        print(f"\n📈 Creating SHAP Visualization Plots...")

        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(24, 18))

        # 1. Summary Plot (Bee Swarm) - Shows feature importance and effects
        print(f"   1/4 Creating summary plot (bee swarm)...")
        ax1 = plt.subplot(2, 2, 1)
        shap.summary_plot(shap_values, X_sample_df, max_display=20, show=False)
        plt.title(f'{model_name} - SHAP Summary Plot\n(Feature Importance & Effects)',
                 fontsize=14, fontweight='bold', pad=20)

        # 2. Bar Plot - Global feature importance
        print(f"   2/4 Creating bar plot (feature importance)...")
        ax2 = plt.subplot(2, 2, 2)
        shap.summary_plot(shap_values, X_sample_df, plot_type="bar",
                         max_display=20, show=False)
        plt.title(f'{model_name} - Global Feature Importance\n(Mean |SHAP Value|)',
                 fontsize=14, fontweight='bold', pad=20)

        # 3. Waterfall plot for a single prediction (first sample)
        print(f"   3/4 Creating waterfall plot (single prediction)...")
        ax3 = plt.subplot(2, 2, 3)
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[0],
            base_values=np.mean(shap_values),
            data=X_sample_df.iloc[0],
            feature_names=feature_names
        ), max_display=15, show=False)
        plt.title(f'{model_name} - Single Prediction Explanation\n(Waterfall Plot for Sample #1)',
                 fontsize=14, fontweight='bold', pad=20)

        # 4. Force plot (for first prediction) - saved separately as it's interactive
        print(f"   4/4 Creating force plot (local explanation)...")
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')

        # Create a text summary of SHAP interpretation
        interpretation_text = f"""
{model_name} SHAP Interpretation Guide
{'=' * 50}

What is SHAP?
SHAP (SHapley Additive exPlanations) uses game
theory to fairly attribute each feature's
contribution to the model's prediction.

How to Read These Plots:

1. Summary Plot (Top Left):
   • Each dot is a feature value for one sample
   • Color shows feature value (red=high, blue=low)
   • X-axis shows SHAP value (impact on prediction)
   • Features sorted by importance (top to bottom)

2. Bar Plot (Top Right):
   • Shows global feature importance
   • Longer bars = more important features
   • Based on mean absolute SHAP value

3. Waterfall Plot (Bottom Left):
   • Explains ONE prediction step-by-step
   • Shows how each feature pushes prediction
   • Starts from base value (mean prediction)
   • Ends at final predicted value

4. Key Insights:
   • Positive SHAP = increases prediction
   • Negative SHAP = decreases prediction
   • Magnitude = strength of effect

Financial Interpretation:
For stock prediction, important features often
include recent price movements, volume changes,
and technical indicators from recent time steps.
        """

        ax4.text(0.05, 0.95, interpretation_text, transform=ax4.transAxes,
                fontsize=10, fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.suptitle(f'{model_name} - Comprehensive SHAP Analysis\n'
                    f'Understanding Model Predictions Through Explainable AI',
                    fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_plots:
            filename = f'{model_name.lower()}_shap_analysis.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved comprehensive analysis to '{filename}'")

        plt.show()

        # Create additional detailed plots
        self._create_detailed_shap_plots(shap_values, X_sample_df, model_name, save_plots)

    def _create_detailed_shap_plots(self, shap_values, X_sample_df, model_name, save_plots):
        """Create additional detailed SHAP plots"""

        print(f"\n📊 Creating Additional Detailed SHAP Plots...")

        # Get top 6 most important features
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-6:][::-1]
        top_features = [X_sample_df.columns[i] for i in top_indices]

        # Create figure with dependence plots for top features
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        print(f"   Creating dependence plots for top 6 features...")

        for idx, (feat_idx, feat_name) in enumerate(zip(top_indices, top_features)):
            ax = axes[idx]
            plt.sca(ax)

            # Create dependence plot
            shap.dependence_plot(
                feat_idx,
                shap_values,
                X_sample_df,
                show=False,
                ax=ax
            )
            ax.set_title(f'Feature: {feat_name}', fontsize=12, fontweight='bold')

        plt.suptitle(f'{model_name} - SHAP Dependence Plots\n'
                    f'How Feature Values Affect Predictions',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_plots:
            filename = f'{model_name.lower()}_shap_dependence.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved dependence plots to '{filename}'")

        plt.show()

    def _calculate_shap_feature_importance(self, shap_values, feature_names):
        """Calculate feature importance from SHAP values"""

        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Create list of (feature_name, importance) tuples
        feature_importance = list(zip(feature_names, mean_abs_shap))

        # Sort by importance (descending)
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        return feature_importance

    def compare_model_interpretability(self, models_to_compare=None, num_samples=300, save_plots=True):
        """
        Compare interpretability across multiple models using SHAP

        Args:
            models_to_compare: List of model names to compare (default: all tree-based models)
            num_samples: Number of samples to use for analysis
            save_plots: Whether to save plots

        Returns:
            dict: Comparison results
        """
        print(f"\n🔬 Comparing Model Interpretability Across Multiple Models...")
        print("=" * 70)

        if models_to_compare is None:
            models_to_compare = ['XGBoost', 'LightGBM', 'RandomForest']

        # Filter to only trained models
        available_models = [m for m in models_to_compare if m in self.trained_models]

        if not available_models:
            print(f"❌ No trained models available for comparison")
            return None

        print(f"   Comparing models: {', '.join(available_models)}")

        # Get SHAP explanations for each model
        all_explanations = {}
        for model_name in available_models:
            print(f"\n   Analyzing {model_name}...")
            explanation = self.explain_model_with_shap(
                model_name,
                num_samples=num_samples,
                save_plots=False  # Don't save individual plots
            )
            if explanation:
                all_explanations[model_name] = explanation

        if not all_explanations:
            print(f"❌ Could not generate explanations for any model")
            return None

        # Create comparison visualization
        self._create_interpretability_comparison_plot(all_explanations, save_plots)

        return all_explanations

    def _create_interpretability_comparison_plot(self, all_explanations, save_plots):
        """Create comparison plot of feature importance across models"""

        print(f"\n📊 Creating Feature Importance Comparison Plot...")

        # Extract top features from each model
        n_top_features = 15
        comparison_data = {}

        for model_name, explanation in all_explanations.items():
            top_features = explanation['feature_importance'][:n_top_features]
            comparison_data[model_name] = {feat: imp for feat, imp in top_features}

        # Get union of all top features
        all_features = set()
        for features_dict in comparison_data.values():
            all_features.update(features_dict.keys())

        # Create DataFrame for comparison
        comparison_df = pd.DataFrame(index=sorted(all_features))
        for model_name, features_dict in comparison_data.items():
            comparison_df[model_name] = comparison_df.index.map(lambda x: features_dict.get(x, 0))

        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Plot 1: Grouped bar chart
        comparison_df.plot(kind='barh', ax=ax1, width=0.8)
        ax1.set_xlabel('Mean |SHAP Value| (Feature Importance)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Features', fontsize=12, fontweight='bold')
        ax1.set_title('Feature Importance Comparison Across Models', fontsize=14, fontweight='bold')
        ax1.legend(title='Models', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='x')

        # Plot 2: Correlation heatmap of feature rankings
        # Convert importance to rankings
        ranking_df = comparison_df.rank(ascending=False)
        correlation = ranking_df.T.corr()

        im = ax2.imshow(correlation, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax2.set_xticks(range(len(correlation.columns)))
        ax2.set_yticks(range(len(correlation.index)))
        ax2.set_xticklabels(correlation.columns, rotation=45, ha='right', fontsize=8)
        ax2.set_yticklabels(correlation.index, fontsize=8)
        ax2.set_title('Feature Ranking Correlation Between Models', fontsize=14, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Correlation', fontsize=10)

        # Add correlation values as text
        for i in range(len(correlation.index)):
            for j in range(len(correlation.columns)):
                text = ax2.text(j, i, f'{correlation.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)

        plt.suptitle('Model Interpretability Comparison\n'
                    'Consistency of Feature Importance Across Models',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        if save_plots:
            filename = 'model_interpretability_comparison.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved comparison to '{filename}'")

        plt.show()

        # Print agreement analysis
        print(f"\n🔍 Feature Importance Agreement Analysis:")
        avg_correlation = correlation.values[np.triu_indices_from(correlation.values, k=1)].mean()
        print(f"   Average feature ranking correlation: {avg_correlation:.3f}")
        if avg_correlation > 0.7:
            print(f"   ✅ High agreement: Models agree on important features")
        elif avg_correlation > 0.4:
            print(f"   ⚠️ Moderate agreement: Some differences in feature importance")
        else:
            print(f"   ❌ Low agreement: Models disagree significantly on features")

    def visualize_optimization_results(self, model_name, save_plots=True):
        """
        Visualize Bayesian optimization results for a specific model

        Args:
            model_name: Name of the model ('XGBoost', 'LightGBM', etc.)
            save_plots: Whether to save plots to disk
        """
        if model_name not in self.optimization_studies:
            print(f"❌ No optimization study found for {model_name}")
            return

        study = self.optimization_studies[model_name]

        print(f"\n📊 Creating Optimization Visualizations for {model_name}...")

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))

        # 1. Optimization History - Manual plot
        ax1 = plt.subplot(2, 2, 1)
        trials = study.trials
        values = [trial.value for trial in trials if trial.value is not None]
        trial_numbers = [trial.number for trial in trials if trial.value is not None]

        ax1.plot(trial_numbers, values, marker='o', linestyle='-', alpha=0.6, label='Trial Value')

        # Add best value line
        best_values = []
        current_best = float('inf')
        for val in values:
            if val < current_best:
                current_best = val
            best_values.append(current_best)
        ax1.plot(trial_numbers, best_values, marker='', linestyle='-', color='red',
                linewidth=2, label='Best Value')

        ax1.set_xlabel('Trial Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Objective Value (MAE)', fontsize=12, fontweight='bold')
        ax1.set_title(f'{model_name} - Optimization History', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Parameter Importances - Manual calculation
        ax2 = plt.subplot(2, 2, 2)
        try:
            importances = optuna.importance.get_param_importances(study)
            params = list(importances.keys())
            importance_values = list(importances.values())

            # Sort by importance
            sorted_indices = np.argsort(importance_values)
            params = [params[i] for i in sorted_indices]
            importance_values = [importance_values[i] for i in sorted_indices]

            ax2.barh(params, importance_values, color='steelblue')
            ax2.set_xlabel('Importance', fontsize=12, fontweight='bold')
            ax2.set_title(f'{model_name} - Hyperparameter Importances', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
        except Exception as e:
            ax2.text(0.5, 0.5, f'Parameter importances\nnot available\n({str(e)})',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'{model_name} - Hyperparameter Importances', fontsize=14, fontweight='bold')

        # 3. Top parameters distribution
        ax3 = plt.subplot(2, 2, 3)
        try:
            # Get top 3 most important parameters
            if importances:
                top_params = list(importances.keys())[:3]

                for param in top_params:
                    param_values = [trial.params[param] for trial in study.trials
                                   if param in trial.params and trial.value is not None]
                    trial_values = [trial.value for trial in study.trials
                                   if param in trial.params and trial.value is not None]

                    ax3.scatter(param_values, trial_values, alpha=0.5,
                              label=param, s=50)

                ax3.set_xlabel('Parameter Value', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Objective Value (MAE)', fontsize=12, fontweight='bold')
                ax3.set_title(f'{model_name} - Top Parameters vs Objective', fontsize=14, fontweight='bold')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Parameter distribution\nnot available',
                        ha='center', va='center', transform=ax3.transAxes)
        except Exception as e:
            ax3.text(0.5, 0.5, f'Parameter distribution\nnot available\n({str(e)})',
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title(f'{model_name} - Top Parameters vs Objective', fontsize=14, fontweight='bold')

        # 4. Trial statistics summary
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')

        # Calculate statistics
        completed_trials = [t for t in study.trials if t.value is not None]
        values_array = np.array([t.value for t in completed_trials])

        stats_text = f"""
        {model_name} Optimization Summary
        {'=' * 40}

        Total Trials: {len(study.trials)}
        Completed Trials: {len(completed_trials)}

        Best Trial: #{study.best_trial.number}
        Best Value (MAE): {study.best_value:.6f}

        Statistics:
          Mean MAE: {np.mean(values_array):.6f}
          Std MAE: {np.std(values_array):.6f}
          Min MAE: {np.min(values_array):.6f}
          Max MAE: {np.max(values_array):.6f}

        Best Parameters:
        """

        for param, value in study.best_params.items():
            if isinstance(value, float):
                stats_text += f"  {param}: {value:.6f}\n        "
            else:
                stats_text += f"  {param}: {value}\n        "

        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=11, fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.suptitle(f'{model_name} Bayesian Optimization Analysis',
                     fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_plots:
            filename = f'{model_name.lower()}_optimization_results.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved to '{filename}'")

        plt.show()

        # Print optimization statistics
        print(f"\n📈 {model_name} Optimization Statistics:")
        print(f"   Total Trials: {len(study.trials)}")
        print(f"   Best Trial: #{study.best_trial.number}")
        print(f"   Best Value (MAE): {study.best_value:.6f}")
        if len(study.trials) > 0 and study.trials[-1].datetime_complete and study.trials[0].datetime_start:
            print(f"   Duration: {study.trials[-1].datetime_complete - study.trials[0].datetime_start}")

    def train_xgboost_model(self, input_length=60, output_length=5, use_optimized_params=False):
        """
        Train XGBoost model for Nikkei prediction

        Args:
            input_length: Number of historical days for input
            output_length: Number of days to predict
            use_optimized_params: If True, use Bayesian-optimized hyperparameters
        """
        print("\n🚀 Training XGBoost Model...")
        print("=" * 50)
        print(f"   Prediction Mode: {self.prediction_mode.upper()}")
        if self.prediction_mode == 'return':
            print(f"   Target: Average return (percentage change)")
        else:
            print(f"   Target: Average close price (¥)")

        # Create sequences
        X_train, y_train = self.create_sequences(self.train_data, input_length, output_length)
        X_val, y_val = self.create_sequences(self.val_data, input_length, output_length)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        self.scalers['XGBoost'] = scaler

        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Feature dimension: {X_train.shape[1]}")

        # Use optimized parameters if available, otherwise use default
        if use_optimized_params and 'XGBoost' in self.best_params:
            print(f"   Using Bayesian-optimized hyperparameters ✨")
            params = self.best_params['XGBoost'].copy()
            params['objective'] = 'reg:squarederror'
            params['random_state'] = 42
            params['n_jobs'] = -1
            model = xgb.XGBRegressor(**params)
        else:
            print(f"   Using default hyperparameters")
            # XGBoost configuration optimized for financial time series
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )

        # Train with early stopping
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )

        # Evaluate
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)

        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)

        print(f"   Training MAE: {train_mae:.6f}")
        print(f"   Validation MAE: {val_mae:.6f}")

        self.trained_models['XGBoost'] = model

        return model

    def train_lightgbm_model(self, input_length=60, output_length=5, use_optimized_params=False):
        """
        Train LightGBM model for Nikkei prediction

        Args:
            input_length: Number of historical days for input
            output_length: Number of days to predict
            use_optimized_params: If True, use Bayesian-optimized hyperparameters
        """
        print("\n🚀 Training LightGBM Model...")
        print("=" * 50)

        # Create sequences
        X_train, y_train = self.create_sequences(self.train_data, input_length, output_length)
        X_val, y_val = self.create_sequences(self.val_data, input_length, output_length)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        self.scalers['LightGBM'] = scaler

        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Feature dimension: {X_train.shape[1]}")

        # Use optimized parameters if available, otherwise use default
        if use_optimized_params and 'LightGBM' in self.best_params:
            print(f"   Using Bayesian-optimized hyperparameters ✨")
            params = self.best_params['LightGBM'].copy()
            params['objective'] = 'regression'
            params['random_state'] = 42
            params['n_jobs'] = -1
            params['verbose'] = -1
            model = lgb.LGBMRegressor(**params)
        else:
            print(f"   Using default hyperparameters")
            # LightGBM configuration optimized for financial time series
            model = lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective='regression',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )

        # Train with early stopping
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        # Evaluate
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)

        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)

        print(f"   Training MAE: {train_mae:.6f}")
        print(f"   Validation MAE: {val_mae:.6f}")

        self.trained_models['LightGBM'] = model

        return model

    def train_random_forest_model(self, input_length=60, output_length=5, use_optimized_params=False):
        """
        Train Random Forest model for Nikkei prediction

        Args:
            input_length: Number of historical days for input
            output_length: Number of days to predict
            use_optimized_params: If True, use Bayesian-optimized hyperparameters
        """
        print("\n🚀 Training Random Forest Model...")
        print("=" * 50)

        # Create sequences
        X_train, y_train = self.create_sequences(self.train_data, input_length, output_length)
        X_val, y_val = self.create_sequences(self.val_data, input_length, output_length)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        self.scalers['RandomForest'] = scaler

        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Feature dimension: {X_train.shape[1]}")

        # Use optimized parameters if available, otherwise use default
        if use_optimized_params and 'RandomForest' in self.best_params:
            print(f"   Using Bayesian-optimized hyperparameters ✨")
            params = self.best_params['RandomForest'].copy()
            params['random_state'] = 42
            params['n_jobs'] = -1
            params['verbose'] = 0
            model = RandomForestRegressor(**params)
        else:
            print(f"   Using default hyperparameters")
            # Random Forest configuration optimized for financial time series
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )

        # Train
        model.fit(X_train_scaled, y_train)

        # Evaluate
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)

        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)

        print(f"   Training MAE: {train_mae:.6f}")
        print(f"   Validation MAE: {val_mae:.6f}")

        self.trained_models['RandomForest'] = model

        return model

    def train_svr_model(self, input_length=60, output_length=5, use_optimized_params=False):
        """
        Train Support Vector Regression model for Nikkei prediction

        Args:
            input_length: Number of historical days for input
            output_length: Number of days to predict
            use_optimized_params: If True, use Bayesian-optimized hyperparameters
        """
        print("\n🚀 Training SVR Model...")
        print("=" * 50)

        # Create sequences
        X_train, y_train = self.create_sequences(self.train_data, input_length, output_length)
        X_val, y_val = self.create_sequences(self.val_data, input_length, output_length)

        # Scale features (critical for SVR)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        self.scalers['SVR'] = scaler

        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Feature dimension: {X_train.shape[1]}")

        # Use optimized parameters if available, otherwise use default
        if use_optimized_params and 'SVR' in self.best_params:
            print(f"   Using Bayesian-optimized hyperparameters ✨")
            params = self.best_params['SVR'].copy()
            params['cache_size'] = 1000
            params['verbose'] = False
            model = SVR(**params)
        else:
            print(f"   Using default hyperparameters")
            # SVR configuration with RBF kernel for non-linear patterns
            model = SVR(
                kernel='rbf',
                C=10.0,
                epsilon=0.01,
                gamma='scale',
                cache_size=1000,
                verbose=False
            )

        # Train
        print("   Training SVR (this may take a while)...")
        model.fit(X_train_scaled, y_train)

        # Evaluate
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)

        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)

        print(f"   Training MAE: {train_mae:.6f}")
        print(f"   Validation MAE: {val_mae:.6f}")

        self.trained_models['SVR'] = model

        return model

    def train_gaussian_process_model(self, input_length=60, output_length=5, use_optimized_params=False, max_samples=1000):
        """
        Train Gaussian Process Regression model for Nikkei prediction

        Args:
            input_length: Number of historical days for input
            output_length: Number of days to predict
            use_optimized_params: If True, use Bayesian-optimized hyperparameters
            max_samples: Maximum training samples to use (default: 1000)
        """
        print("\n🚀 Training Gaussian Process Regression Model...")
        print("=" * 50)

        # Create sequences
        X_train, y_train = self.create_sequences(self.train_data, input_length, output_length)
        X_val, y_val = self.create_sequences(self.val_data, input_length, output_length)

        # For GPR, use a subset of training data due to computational complexity
        # Sample strategically: more recent data + random samples
        if len(X_train) > max_samples:
            # Take last half (most recent) + random samples from earlier
            recent_size = max_samples // 2
            early_size = max_samples - recent_size
            recent_indices = np.arange(len(X_train) - recent_size, len(X_train))
            early_indices = np.random.choice(len(X_train) - recent_size, size=early_size, replace=False)
            sample_indices = np.concatenate([early_indices, recent_indices])
            X_train_sampled = X_train[sample_indices]
            y_train_sampled = y_train[sample_indices]
        else:
            X_train_sampled = X_train
            y_train_sampled = y_train

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sampled)
        X_val_scaled = scaler.transform(X_val)

        self.scalers['GPR'] = scaler

        print(f"   Training samples: {len(X_train_scaled)} (sampled from {len(X_train)})")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Feature dimension: {X_train.shape[1]}")

        # Use optimized parameters if available, otherwise use default
        if use_optimized_params and 'GPR' in self.best_params:
            print(f"   Using Bayesian-optimized hyperparameters ✨")
            best_params = self.best_params['GPR']

            # Reconstruct kernel from optimized parameters
            constant_kernel = ConstantKernel(
                constant_value=best_params.get('constant_value', 1.0),
                constant_value_bounds=(
                    best_params.get('constant_bounds_low', 1e-3),
                    best_params.get('constant_bounds_high', 1e3)
                )
            )

            kernel_type = best_params.get('kernel_type', 'RBF')
            length_scale = best_params.get('length_scale', 1.0)
            length_scale_bounds = (
                best_params.get('length_scale_bounds_low', 1e-2),
                best_params.get('length_scale_bounds_high', 1e2)
            )

            if kernel_type == 'RBF':
                main_kernel = RBF(
                    length_scale=length_scale,
                    length_scale_bounds=length_scale_bounds
                )
            elif kernel_type == 'Matern':
                from sklearn.gaussian_process.kernels import RationalQuadratic
                nu = best_params.get('matern_nu', 1.5)
                main_kernel = Matern(
                    length_scale=length_scale,
                    length_scale_bounds=length_scale_bounds,
                    nu=nu
                )
            elif kernel_type == 'RationalQuadratic':
                from sklearn.gaussian_process.kernels import RationalQuadratic
                alpha_rq = best_params.get('alpha_rq', 1.0)
                main_kernel = RationalQuadratic(
                    length_scale=length_scale,
                    alpha=alpha_rq,
                    length_scale_bounds=length_scale_bounds
                )
            else:  # RBF+Matern
                nu = best_params.get('matern_nu', 1.5)
                rbf_kernel = RBF(
                    length_scale=length_scale,
                    length_scale_bounds=length_scale_bounds
                )
                matern_kernel = Matern(
                    length_scale=length_scale * 0.5,
                    length_scale_bounds=length_scale_bounds,
                    nu=nu
                )
                main_kernel = rbf_kernel + matern_kernel

            white_kernel = WhiteKernel(
                noise_level=best_params.get('noise_level', 1e-5),
                noise_level_bounds=(
                    best_params.get('noise_bounds_low', 1e-10),
                    best_params.get('noise_bounds_high', 1e-1)
                )
            )

            kernel = constant_kernel * main_kernel + white_kernel

            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=best_params.get('alpha', 1e-6),
                n_restarts_optimizer=best_params.get('n_restarts_optimizer', 5),
                normalize_y=best_params.get('normalize_y', True),
                random_state=42
            )

            print(f"   Kernel type: {kernel_type}")
            print(f"   Alpha: {best_params.get('alpha', 1e-6):.2e}")
            print(f"   Normalize Y: {best_params.get('normalize_y', True)}")

        else:
            print(f"   Using default hyperparameters")
            # GPR configuration with RBF kernel + noise (default)
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + \
                     WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))

            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                n_restarts_optimizer=5,
                normalize_y=True,
                random_state=42
            )

        # Train
        print("   Training GPR (this may take a while)...")
        model.fit(X_train_scaled, y_train_sampled)

        # Evaluate
        train_pred = model.predict(X_train_scaled)
        val_pred, val_std = model.predict(X_val_scaled, return_std=True)

        train_mae = mean_absolute_error(y_train_sampled, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)

        print(f"   Training MAE: {train_mae:.6f}")
        print(f"   Validation MAE: {val_mae:.6f}")
        print(f"   Avg Uncertainty (std): {np.mean(val_std):.6f}")

        self.trained_models['GPR'] = model

        return model

    def train_ensemble_model(self, models_to_ensemble=None):
        """Create ensemble model combining predictions from multiple ML models"""
        print("\n🚀 Creating Ensemble Model...")
        print("=" * 50)

        if models_to_ensemble is None:
            models_to_ensemble = ['XGBoost', 'LightGBM', 'RandomForest']

        # Check if all required models are trained
        missing_models = [m for m in models_to_ensemble if m not in self.trained_models]
        if missing_models:
            print(f"   ⚠️ Missing models: {missing_models}")
            print(f"   Training missing models first...")

            for model_name in missing_models:
                if model_name == 'XGBoost':
                    self.train_xgboost_model()
                elif model_name == 'LightGBM':
                    self.train_lightgbm_model()
                elif model_name == 'RandomForest':
                    self.train_random_forest_model()

        # Store ensemble configuration
        self.trained_models['Ensemble'] = {
            'models': models_to_ensemble,
            'weights': None  # Equal weights initially
        }

        print(f"   Ensemble created with models: {models_to_ensemble}")
        print(f"   Using equal weighting strategy")

        return self.trained_models['Ensemble']

    def train_all_models(self, input_length=60, output_length=5):
        """Train all ML models"""
        print("\n🚀 Training All ML Models...")
        print("=" * 60)

        # Train each model
        self.train_xgboost_model(input_length, output_length)
        self.train_lightgbm_model(input_length, output_length)
        self.train_random_forest_model(input_length, output_length)
        self.train_svr_model(input_length, output_length)
        self.train_gaussian_process_model(input_length, output_length)

        # Create ensemble
        self.train_ensemble_model(['XGBoost', 'LightGBM', 'RandomForest'])

        print("\n✅ All ML models trained successfully!")

        return self.trained_models

    def generate_ml_predictions(self, model_name, data=None, data_type=None,
                                prediction_days=60, input_length=60, output_length=5):
        """Generate predictions for a specific ML model (following LSTM format)"""
        if model_name not in self.trained_models:
            print(f"⚠️ Model {model_name} not found in trained models.")
            return None

        # Handle data parameter
        if data is None:
            if data_type == 'test':
                data = self.test_data
            elif data_type == 'train':
                data = self.train_data
            else:
                print(f"⚠️ Must provide either 'data' or 'data_type' parameter.")
                return None

        if len(data) < input_length + prediction_days:
            print(f"⚠️ Insufficient data for predictions.")
            return None

        # Get the model and scaler
        model = self.trained_models[model_name]
        scaler = self.scalers.get(model_name)

        if scaler is None and model_name != 'Ensemble':
            print(f"⚠️ Scaler not found for {model_name}")
            return None

        predictions = []
        uncertainties = []
        dates = []
        actual_prices = []

        # Generate predictions for the last prediction_days
        # CRITICAL: Exclude 'close' from input features (same as training)
        feature_cols = [col for col in data.columns if col not in ['date', 'close']]

        # We want to generate predictions for the last 'prediction_days' days
        # For each prediction at day i, we use [i-input_length:i] as input
        # and compare with actual prices at [i:i+output_length]

        # Calculate valid range for making predictions
        # Minimum start: we need at least input_length days before
        min_start = input_length
        # Maximum end: we need at least output_length days after for validation
        max_end = len(data) - output_length

        # Calculate the actual range we want to predict
        # We want the last 'prediction_days' valid prediction points
        start_idx = max(min_start, max_end - prediction_days)
        end_idx = max_end

        for i in range(start_idx, end_idx):
            # i is the prediction point
            # Use data[i-input_length:i] as input
            # Predict for data[i:i+output_length]

            # Create input sequence from past data
            X_seq = data[feature_cols].iloc[i-input_length:i].values.flatten()
            X_scaled = scaler.transform([X_seq])

            # Get current price and date (at prediction point i-1, last day of input)
            current_price = data['close'].iloc[i-1]
            dates.append(data.index[i])

            # Define actual target based on prediction mode (must match training!)
            if self.prediction_mode == 'return':
                # Actual: average return over next output_length days
                future_prices = data['close'].iloc[i:i+output_length].values
                actual_return = np.mean((future_prices - current_price) / current_price)
                # For visualization, convert to price
                actual_price = current_price * (1 + actual_return)
            else:  # prediction_mode == 'price'
                # Actual: NEXT close price (at time i)
                actual_price = data['close'].iloc[i]

            actual_prices.append(actual_price)

            # Get model prediction (return or price depending on mode)
            if model_name == 'Ensemble':
                # Ensemble prediction
                ensemble_preds = []
                for sub_model_name in model['models']:
                    sub_model = self.trained_models[sub_model_name]
                    sub_scaler = self.scalers[sub_model_name]
                    X_sub_scaled = sub_scaler.transform([X_seq])
                    pred_value = sub_model.predict(X_sub_scaled)[0]
                    ensemble_preds.append(pred_value)

                pred_value = np.mean(ensemble_preds)
                uncertainty = np.std(ensemble_preds)
            elif model_name == 'GPR':
                pred_value, uncertainty = model.predict(X_scaled, return_std=True)
                pred_value = pred_value[0]
                uncertainty = uncertainty[0]
            else:
                pred_value = model.predict(X_scaled)[0]
                uncertainty = 0.01  # Default uncertainty for non-GPR models

            # Convert prediction to price based on prediction mode
            if self.prediction_mode == 'return':
                # Model predicts return, convert to price
                pred_price = current_price * (1 + pred_value)
                unc_price = abs(uncertainty * current_price)
            else:  # prediction_mode == 'price'
                # Model directly predicts the next close price
                pred_price = pred_value
                unc_price = uncertainty

            predictions.append(pred_price)
            uncertainties.append(unc_price)

        return {
            'predictions': np.array(predictions),
            'uncertainties': np.array(uncertainties),
            'dates': dates,
            'actual_prices': np.array(actual_prices),
            'model_name': model_name,
            'prediction_type': data_type if data_type else 'historical'
        }

    def _calculate_performance_metrics(self, predictions, actual_prices):
        """Calculate common performance metrics (same as LSTM)"""
        pred_array = np.array(predictions)
        actual_array = np.array(actual_prices)

        mae = np.mean(np.abs(pred_array - actual_array))
        mse = np.mean((pred_array - actual_array) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual_array - pred_array) / (actual_array + 1e-8))) * 100

        # Calculate directional accuracy
        if len(predictions) > 1:
            pred_changes = np.sign(np.diff(pred_array))
            actual_changes = np.sign(np.diff(actual_array))
            directional_accuracy = np.mean(pred_changes == actual_changes) * 100
        else:
            directional_accuracy = 0.0

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }

    def _create_candlestick_chart(self, ax, chart_data):
        """Create candlestick chart with consistent styling (same as LSTM)"""
        dates = chart_data.index
        opens = chart_data['open'].values
        highs = chart_data['high'].values
        lows = chart_data['low'].values
        closes = chart_data['close'].values

        for i, (date, open_price, high, low, close) in enumerate(zip(dates, opens, highs, lows, closes)):
            color = '#26A69A' if close >= open_price else '#EF5350'
            edge_color = '#004D40' if close >= open_price else '#B71C1C'

            body_height = abs(close - open_price)
            body_bottom = min(close, open_price)

            rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                           facecolor=color, alpha=0.8, edgecolor=edge_color, linewidth=1)
            ax.add_patch(rect)
            ax.plot([i, i], [low, high], color='black', linewidth=1.5, alpha=0.7)

        return dates

    def plot_ml_test_predictions(self, model_name=None, prediction_days=50, save_plots=True):
        """
        Plot ML model predictions following exact format of LSTM test predictions
        """
        if model_name is None:
            # Use XGBoost as default
            model_name = 'XGBoost'

        if model_name not in self.trained_models:
            print(f"❌ Model {model_name} not found.")
            return None

        print(f"📈 Creating {model_name} Test Data Predictions Chart...")

        # Generate predictions for test data
        test_predictions = self.generate_ml_predictions(
            model_name, data_type='test', prediction_days=prediction_days
        )

        if test_predictions is None:
            print(f"❌ Cannot generate {model_name} test predictions.")
            return None

        # Create figure - exactly following LSTM format
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))

        # Main title with prediction mode
        mode_text = "Return Prediction" if self.prediction_mode == 'return' else "Direct Price Prediction"
        fig.suptitle(f'{model_name} Model - Test Data Performance\n{mode_text}',
                    fontsize=16, fontweight='bold', y=0.95)

        # Get test data for candlestick chart
        chart_data = self.test_data.tail(prediction_days)

        # Plot candlestick chart
        dates = chart_data.index
        opens = chart_data['open'].values
        highs = chart_data['high'].values
        lows = chart_data['low'].values
        closes = chart_data['close'].values

        # Create candlestick chart
        for i, (date, open_price, high, low, close) in enumerate(zip(dates, opens, highs, lows, closes)):
            color = '#26A69A' if close >= open_price else '#EF5350'
            edge_color = '#004D40' if close >= open_price else '#B71C1C'

            body_height = abs(close - open_price)
            body_bottom = min(close, open_price)

            rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                           facecolor=color, alpha=0.8, edgecolor=edge_color, linewidth=1)
            ax.add_patch(rect)
            ax.plot([i, i], [low, high], color='black', linewidth=1.5, alpha=0.7)

        # Overlay predictions
        num_preds = len(test_predictions['predictions'])
        chart_len = len(dates)

        # Align predictions with the chart data
        start_idx = max(0, chart_len - num_preds)
        pred_indices = list(range(start_idx, chart_len))

        # Take only the predictions that fit within our chart
        pred_to_show = min(num_preds, chart_len)
        pred_values = test_predictions['predictions'][-pred_to_show:]
        actual_values = test_predictions['actual_prices'][-pred_to_show:]
        uncertainty_values = test_predictions['uncertainties'][-pred_to_show:]

        # Adjust indices to match
        pred_indices = pred_indices[-len(pred_values):]

        if pred_indices:
            # Plot predictions
            ax.plot(pred_indices, pred_values,
                   color='blue', marker='o', linewidth=4, markersize=8,
                   label=f'{model_name} Predictions', alpha=0.9,
                   markerfacecolor='blue', markeredgecolor='white', markeredgewidth=2)

            # Add uncertainty bands
            uncertainties = np.array(uncertainty_values)
            upper_band = np.array(pred_values) + uncertainties
            lower_band = np.array(pred_values) - uncertainties
            ax.fill_between(pred_indices, lower_band, upper_band,
                           alpha=0.2, color='blue', label='Uncertainty Band')

            # Calculate performance metrics
            mae = np.mean(np.abs(np.array(pred_values) - np.array(actual_values)))
            mape = np.mean(np.abs((np.array(actual_values) - np.array(pred_values)) / np.array(actual_values))) * 100
            rmse = np.sqrt(np.mean((np.array(pred_values) - np.array(actual_values))**2))

            # Calculate directional accuracy
            pred_changes = np.diff(pred_values)
            actual_changes = np.diff(actual_values)
            directional_accuracy = np.mean(np.sign(pred_changes) == np.sign(actual_changes)) * 100

            # Add performance metrics box (same format as LSTM)
            mode_info = "Predicting: Returns" if self.prediction_mode == 'return' else "Predicting: Next Price"
            feature_info = "Features: Exclude 'close'"
            metrics_text = f"""Test Performance:
{mode_info}
MAPE: {mape:.1f}%
Directional: {directional_accuracy:.1f}%"""

            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   fontsize=12, fontweight='bold', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

        # Format chart (exactly like LSTM)
        ax.set_title(f'{model_name} Test Predictions', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price (¥)', fontsize=12)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Format x-axis
        step = max(1, len(dates) // 10)
        tick_positions = range(0, len(dates), step)
        tick_labels = [dates[i].strftime('%m/%d/%y') for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45)

        plt.tight_layout()

        if save_plots:
            plt.savefig(f'{model_name.lower()}_test_predictions.png', dpi=300, bbox_inches='tight')
            print(f"📊 {model_name} test predictions saved to '{model_name.lower()}_test_predictions.png'")

        plt.show()

        # Print test statistics
        if pred_indices:
            print(f"\n📊 {model_name.upper()} TEST PERFORMANCE:")
            print(f"   Mean Absolute Error: ¥{mae:.0f}")
            print(f"   Mean Absolute Percentage Error: {mape:.1f}%")
            print(f"   Root Mean Square Error: ¥{rmse:.0f}")
            print(f"   Directional Accuracy: {directional_accuracy:.1f}%")
            print(f"   Total Predictions: {len(pred_values)}")

        return fig

    def plot_all_models_comparison(self, prediction_days=50, save_plots=True):
        """
        Plot comparison of all ML models on test data
        """
        print("📈 Creating All Models Comparison Chart...")

        # Select models to compare
        models_to_compare = ['XGBoost', 'LightGBM', 'RandomForest', 'SVR', 'Ensemble']

        # Create figure with subplots
        n_models = len(models_to_compare)
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.flatten()

        fig.suptitle('Machine Learning Models - Comprehensive Performance Comparison',
                    fontsize=20, fontweight='bold', y=0.98)

        # Plot each model
        for idx, model_name in enumerate(models_to_compare):
            if model_name not in self.trained_models:
                continue

            ax = axes[idx]

            # Generate predictions
            test_predictions = self.generate_ml_predictions(
                model_name, data_type='test', prediction_days=prediction_days
            )

            if test_predictions is None:
                continue

            # Get test data for candlestick chart
            chart_data = self.test_data.tail(prediction_days)
            dates = chart_data.index

            # Simplified candlestick (smaller for subplot)
            closes = chart_data['close'].values
            ax.plot(range(len(closes)), closes, color='gray', alpha=0.5, linewidth=2, label='Actual Price')

            # Overlay predictions
            pred_values = test_predictions['predictions']
            actual_values = test_predictions['actual_prices']

            # Align
            num_preds = len(pred_values)
            chart_len = len(dates)
            start_idx = max(0, chart_len - num_preds)
            pred_indices = list(range(start_idx, chart_len))[-len(pred_values):]

            # Plot
            ax.plot(pred_indices, pred_values,
                   color='blue', marker='o', linewidth=3, markersize=6,
                   label=f'{model_name}', alpha=0.9)

            # Metrics
            mae = np.mean(np.abs(pred_values - actual_values))
            mape = np.mean(np.abs((actual_values - pred_values) / actual_values)) * 100

            if len(pred_values) > 1:
                pred_changes = np.sign(np.diff(pred_values))
                actual_changes = np.sign(np.diff(actual_values))
                directional = np.mean(pred_changes == actual_changes) * 100
            else:
                directional = 0

            # Metrics box
            metrics_text = f"""MAE: ¥{mae:.0f}
MAPE: {mape:.1f}%
Dir: {directional:.1f}%"""

            ax.text(0.03, 0.97, metrics_text, transform=ax.transAxes,
                   fontsize=10, fontweight='bold', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

            ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
            ax.set_ylabel('Price (¥)', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        # Hide unused subplot
        if n_models < len(axes):
            for idx in range(n_models, len(axes)):
                axes[idx].axis('off')

        plt.tight_layout()

        if save_plots:
            plt.savefig('all_ml_models_comparison.png', dpi=300, bbox_inches='tight')
            print("📊 All models comparison saved to 'all_ml_models_comparison.png'")

        plt.show()

        return fig

    def save_trained_models(self, save_dir="saved_ml_models", include_metadata=True):
        """Save all trained ML models to disk"""
        if not self.trained_models:
            print("❌ No trained models to save.")
            return None

        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"💾 Saving trained ML models to '{save_dir}'...")

        saved_info = {
            'timestamp': timestamp,
            'models': {},
            'scalers': {}
        }

        # Save models and scalers
        for model_name in self.trained_models.keys():
            if model_name == 'Ensemble':
                # Save ensemble configuration
                ensemble_filename = f"ensemble_{timestamp}.pkl"
                ensemble_path = os.path.join(save_dir, ensemble_filename)
                with open(ensemble_path, 'wb') as f:
                    pickle.dump(self.trained_models[model_name], f)
                saved_info['models'][model_name] = ensemble_filename
            else:
                # Save model
                model_filename = f"{model_name.lower()}_{timestamp}.pkl"
                model_path = os.path.join(save_dir, model_filename)
                with open(model_path, 'wb') as f:
                    pickle.dump(self.trained_models[model_name], f)
                saved_info['models'][model_name] = model_filename

                # Save scaler
                if model_name in self.scalers:
                    scaler_filename = f"{model_name.lower()}_scaler_{timestamp}.pkl"
                    scaler_path = os.path.join(save_dir, scaler_filename)
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scalers[model_name], f)
                    saved_info['scalers'][model_name] = scaler_filename

            print(f"  ✅ {model_name}")

        # Save metadata
        if include_metadata:
            metadata_filename = f"ml_models_metadata_{timestamp}.json"
            metadata_path = os.path.join(save_dir, metadata_filename)

            metadata = {
                'saved_info': saved_info,
                'data_info': {
                    'train_period': f"{self.train_data.index[0]} to {self.train_data.index[-1]}",
                    'test_period': f"{self.test_data.index[0]} to {self.test_data.index[-1]}",
                    'train_samples': len(self.train_data),
                    'test_samples': len(self.test_data)
                }
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            print(f"  ✅ Metadata → {metadata_filename}")

        print(f"📁 ML Models saved successfully!")
        print(f"   Directory: {save_dir}")
        print(f"   Models: {len(saved_info['models'])}")

        return saved_info


##===================================================================================
## Main Execution
##===================================================================================


print("🚀 MACHINE LEARNING MODELS FOR NIKKEI INDEX PREDICTION")
print("=" * 70)
print("State-of-the-art ML models based on 2024-2025 research")
print("Models: XGBoost, LightGBM, Random Forest, SVR, Gaussian Process")
print("=" * 70)

# Configuration
config = {
    'start_date': "2000-01-01",
    'end_date': "2025-03-31",
    'input_length': 60,
    'output_length': 5,
    'prediction_days': 60
}

# Initialize
ml_comparison = MLModelsComparison(prediction_mode='return')

# Prepare data
data = ml_comparison.prepare_data(
    start_date=config['start_date'],
    end_date=config['end_date']
)

#-------------------------------------------------------------------------------

print("\n" + "=" * 80)
print("BAYESIAN OPTIMIZATION EXAMPLE - Uncomment to run")
print("=" * 80)


# EXAMPLE 1: Optimize and train XGBoost with best parameters
# Step 1: Optimize hyperparameters (this will take some time)
best_xgb_params = ml_comparison.optimize_xgboost_hyperparameters(
    input_length=60,
    output_length=5,
    n_trials=100  # Increase for better results (e.g., 200-300)
)

# Step 2: Visualize optimization results
ml_comparison.visualize_optimization_results('XGBoost', save_plots=True)

# Step 3: Train model with optimized parameters
xgboost_model_optimized = ml_comparison.train_xgboost_model(
    input_length=60,
    output_length=5,
    use_optimized_params=True  # Use Bayesian-optimized parameters
)

# Generate comprehensive SHAP analysis for XGBoost
shap_results = ml_comparison.explain_model_with_shap(
    model_name='XGBoost',
    num_samples=500,  # Number of samples to analyze
    save_plots=False   # Save visualizations to disk
)

# Custom force plot for specific sample
sample_idx = 0
shap.force_plot(
    base_value=shap_results['explainer'].expected_value,
    shap_values=shap_results['shap_values'][sample_idx],
    features=shap_results['X_sample'].iloc[sample_idx],
    matplotlib=True
)
plt.show()

# Custom dependence plot with specific interaction
shap.dependence_plot(
    'close_t-1',
    shap_results['shap_values'],
    shap_results['X_sample'],
    interaction_index='volume_t-1'
)

# Step 4: Compare predictions
ml_comparison.plot_ml_test_predictions(model_name='XGBoost', prediction_days=50, save_plots=True)

# -------------------------------------------------------------------------------

# EXAMPLE 2: Optimize and train LightGBM
best_lgb_params = ml_comparison.optimize_lightgbm_hyperparameters(
    input_length=60,
    output_length=5,
    n_trials=100
)

ml_comparison.visualize_optimization_results('LightGBM', save_plots=True)

lightgbm_model_optimized = ml_comparison.train_lightgbm_model(
    input_length=60,
    output_length=5,
    use_optimized_params=True
)

ml_comparison.plot_ml_test_predictions(model_name='LightGBM', prediction_days=50, save_plots=True)

# -------------------------------------------------------------------------------

# EXAMPLE 3: Optimize Random Forest
best_rf_params = ml_comparison.optimize_randomforest_hyperparameters(
    input_length=60,
    output_length=5,
    n_trials=100
)

ml_comparison.visualize_optimization_results('RandomForest', save_plots=True)

rf_model_optimized = ml_comparison.train_random_forest_model(
    input_length=60,
    output_length=5,
    use_optimized_params=False
)

ml_comparison.plot_ml_test_predictions(model_name='RandomForest', prediction_days=50, save_plots=True)

# -------------------------------------------------------------------------------

# EXAMPLE 4: Optimize SVR (fewer trials due to computational cost)

best_gaussian_params = ml_comparison.optimize_gaussian_process_hyperparameters(
    input_length=60,
    output_length=5,
    max_samples=1000,
    n_trials=100  # Fewer trials for SVR
)

ml_comparison.visualize_optimization_results('GPR', save_plots=True)

gaussian_model_optimized = ml_comparison.train_gaussian_process_model(
    input_length=60,
    output_length=5,
    use_optimized_params=True
)

ml_comparison.plot_ml_test_predictions(model_name='GPR', prediction_days=50, save_plots=True)

# -------------------------------------------------------------------------------

# EXAMPLE 5: Save optimized parameters for future use
import json

optimized_params = {
    'XGBoost': ml_comparison.best_params.get('XGBoost', {}),
    'LightGBM': ml_comparison.best_params.get('LightGBM', {}),
    'RandomForest': ml_comparison.best_params.get('RandomForest', {}),
    'SVR': ml_comparison.best_params.get('SVR', {})
}

with open('optimized_hyperparameters.json', 'w') as f:
    json.dump(optimized_params, f, indent=2)

print("✅ Optimized hyperparameters saved to 'optimized_hyperparameters.json'")

# -------------------------------------------------------------------------------

# EXAMPLE 6: Compare all models (default vs optimized)
ml_comparison.plot_all_models_comparison(prediction_days=50, save_plots=True)






























































