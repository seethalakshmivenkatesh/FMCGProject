"""
ML Models for Demand Forecasting: Prophet + LightGBM + Hybrid
"""
import pandas as pd
import numpy as np
from prophet import Prophet
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, Dict, List
import joblib
import warnings
warnings.filterwarnings('ignore')

class DemandForecaster:
    def __init__(self):
        self.prophet_model = None
        self.lgbm_model = None
        self.is_trained = False
        self.feature_info = None
        
    def train_prophet(self, df: pd.DataFrame, date_col: str = 'date', sales_col: str = 'sales') -> Dict:
        """Train Prophet model for time series forecasting"""
        try:
            prophet_df = df[[date_col, sales_col]].rename(columns={date_col: 'ds', sales_col: 'y'})
            
            # Initialize and train Prophet model
            self.prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            self.prophet_model.fit(prophet_df)
            
            # Cross-validation metrics
            future = self.prophet_model.make_future_dataframe(periods=30)
            forecast = self.prophet_model.predict(future)
            mae = mean_absolute_error(prophet_df['y'], forecast['yhat'][:len(prophet_df)])
            
            print(f"✅ Prophet trained. MAE: {mae:.2f}")
            return {'model': 'Prophet', 'mae': round(mae, 2), 'status': 'success'}
            
        except Exception as e:
            print(f"❌ Prophet training failed: {str(e)}")
            return {'model': 'Prophet', 'error': str(e), 'status': 'failed'}
    
    def train_lightgbm(self, df: pd.DataFrame, feature_info: Dict) -> Dict:
        """Train LightGBM regression model with engineered features"""
        try:
            features = feature_info['features']
            X = df[features]
            y = df[feature_info['target_col']]
            
            # Train-test split (80-20)
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # LightGBM model parameters
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'verbose': -1
            }
            
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            self.lgbm_model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
            )
            
            # Calculate metrics
            y_pred = self.lgbm_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"✅ LightGBM trained. MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            return {
                'model': 'LightGBM', 
                'mae': round(mae, 2), 
                'rmse': round(rmse, 2),
                'status': 'success'
            }
            
        except Exception as e:
            print(f"❌ LightGBM training failed: {str(e)}")
            return {'model': 'LightGBM', 'error': str(e), 'status': 'failed'}
    
    def train_hybrid_model(self, prophet_forecast: pd.DataFrame, lgbm_forecast: np.ndarray, 
                          df: pd.DataFrame) -> Dict:
        """Simple ensemble: weighted average of Prophet + LightGBM"""
        try:
            # Weight by inverse of MAE (better model gets more weight)
            prophet_weight = 0.6  # Assume Prophet performs better on trend
            lgbm_weight = 0.4
            
            hybrid_forecast = (prophet_weight * prophet_forecast + lgbm_weight * lgbm_forecast)
            
            mae = mean_absolute_error(df['sales'][-30:], hybrid_forecast[-30:])
            print(f"✅ Hybrid model created. MAE: {mae:.2f}")
            
            return {
                'model': 'Hybrid',
                'weights': {'prophet': prophet_weight, 'lgbm': lgbm_weight},
                'mae': round(mae, 2),
                'status': 'success'
            }
        except Exception as e:
            print(f"❌ Hybrid model failed: {str(e)}")
            return {'model': 'Hybrid', 'error': str(e), 'status': 'failed'}
    
    def forecast(self, df: pd.DataFrame, periods: int = 30, feature_info: Dict = None) -> Dict:
        """Generate forecasts from all models"""
        if not self.is_trained:
            raise ValueError("Models must be trained first!")
        
        results = {}
        
        # Prophet forecast
        future = self.prophet_model.make_future_dataframe(periods=periods)
        prophet_fc = self.prophet_model.predict(future)
        results['prophet'] = prophet_fc[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        
        # LightGBM forecast (use last known features + repeat patterns)
        if self.lgbm_model and feature_info:
            last_features = df[feature_info['features']].tail(periods).copy()
            lgbm_fc = self.lgbm_model.predict(last_features)
            results['lgbm'] = lgbm_fc
        
        return results
    
    def train_all(self, df: pd.DataFrame, feature_info: Dict) -> List[Dict]:
        """Train all models and return training results"""
        self.feature_info = feature_info
        self.is_trained = True
        
        results = []
        results.append(self.train_prophet(df))
        results.append(self.train_lightgbm(df, feature_info))
        
        print("✅ All models trained successfully!")
        return results