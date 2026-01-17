# src/core/week_3/preparing/data_preparer.py

"""
Professional data preparation class for retail demand forecasting.
Handles feature engineering, scaling, NaN imputation, train/test split,
sequence building for LSTM, and integrates with DataLoader for direct dataset access.
"""

import os  # noqa: F401
import pandas as pd
import numpy as np
from typing import List, Tuple

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils.utils import get_path  # noqa: F401
from src.core.week_1 import DataLoader  # type: ignore #

class DataPreparer:
    def __init__(self, 
                 df: pd.DataFrame,  
                 filename: str = "final_train_dataset.csv", 
                 week: int = 1, 
                 max_date: str = "2014-04-01",
                 load_mode: str = "date_filter",
                 filter_folder: str = "filtered",
                 folder_name: str = "features",
                 store_ids: List[int] = [24],
                 item_ids: List[int] = [105577],
            ):
        print("Initializing DataPreparer...")
        # Loader integration
        self.filename = filename
        self.week = week if week == 1 else 1
        self.load_mode = load_mode
        self.filter_folder = filter_folder
        self.folder_name = folder_name
        self.store_ids = store_ids
        self.item_ids = item_ids
        # Cutoff date
        self.max_date = pd.to_datetime(max_date)
        print(f"Max date set to: {self.max_date}")
        
        self.df = df if df is not None else self._load_data()

        

        # Ensure datetime + CV group
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['cv_group'] = self.df['date'].dt.to_period('M').astype(str)
        print(f"Data shape after initialization: {self.df.shape}")
        print(f"Data columns: {self.df.columns.to_list()}.")

    def _load_data(self):
        print("Loading data from DataLoader...")
        loader = DataLoader()
        if self.load_mode == "date_filter":
            print("📌 Using loader: load_data_filtered_by_date()")

            df = loader.load_data_filtered_by_date(
                folder_name=self.folder_name,
                table_name=self.filename,
                max_date=self.max_date,
                filter_folder=self.filter_folder,
                week=self.week,
                force_recompute=False
            )

        elif self.load_mode == "csv_filter":
            print("📌 Using loader: load_filtered_csv()")

            filters = {
                "MAX_DATE": self.max_date,
                "STORE_IDS": self.store_ids,
                "ITEM_IDS": self.item_ids
            }

            df = loader.load_filtered_csv(
                folder_name=self.folder_name,
                table_name=self.filename,
                filters=filters,
                filter_folder=self.filter_folder,
                week=self.week,
                force_recompute=False
            )
        print(f"Data loaded with shape: {df.shape}")
        return df

    # -----------------------------
    # Feature engineering
    # -----------------------------
    def engineer_features(self) -> None:
        """
        Engineer features for the dataset.
        This method adds lag features, rolling statistics, and other useful features for time series forecasting.
        """
        print("🔧 Starting feature engineering process...")

        # Create a copy of the original dataframe to avoid modifying it directly
        df = self.df.copy()
        print(f"📌 Original data shape: {df.shape}")

        # Filter data based on max_date to ensure we're only working with relevant historical data
        df = df[df['date'] < self.max_date]
        print(f"📅 Data filtered by max_date: {self.max_date}")
        print(f"📊 Data shape after date filter: {df.shape}")

        # Ensure 'onpromotion' is binary (0 or 1) for consistency in modeling
        if 'onpromotion' in df.columns:
            print("🔄 Converting 'onpromotion' to binary (0/1)...")
            print(f"   Before conversion: {df['onpromotion'].unique()}")
            df['onpromotion'] = df['onpromotion'].apply(lambda x: 1 if x is True else 0)
            print(f"   After conversion: {df['onpromotion'].unique()}")

        # Add lag features to capture temporal patterns
        print("⏳ Adding lag features for unit_sales...")
        for lag in [1, 7, 30]:
            col = f"unit_sales_lag_{lag}"
            if col not in df.columns:
                print(f"   Adding {col} feature...")
                df[col] = df.groupby(['store_nbr', 'item_nbr'])['unit_sales'].shift(lag)
                print(f"   ✓ {col} added with {df[col].isna().sum()} NaN values")

        # Add rolling statistics to capture trends and seasonality
        print("📈 Adding rolling statistics features...")
        if 'unit_sales_r7_mean' not in df.columns:
            print("   Adding unit_sales_r7_mean (7-day rolling mean)...")
            df['unit_sales_r7_mean'] = (
                df.groupby(['store_nbr', 'item_nbr'])['unit_sales']
                .shift(1).rolling(window=7).mean().reset_index(level=0, drop=True)
            )
            print(f"   ✓ unit_sales_r7_mean added with {df['unit_sales_r7_mean'].isna().sum()} NaN values")

        if 'unit_sales_r7_std' not in df.columns:
            print("   Adding unit_sales_r7_std (7-day rolling std)...")
            df['unit_sales_r7_std'] = (
                df.groupby(['store_nbr', 'item_nbr'])['unit_sales']
                .shift(1).rolling(window=7).std().reset_index(level=0, drop=True)
            )
            print(f"   ✓ unit_sales_r7_std added with {df['unit_sales_r7_std'].isna().sum()} NaN values")

        # Add days since last sale to capture product availability patterns
        if 'days_since_last_sale' not in df.columns:
            print("📅 Adding days_since_last_sale feature...")
            df['days_since_last_sale'] = (
                df.groupby(['store_nbr', 'item_nbr'])['unit_sales']
                .transform(lambda x: x.eq(0).cumsum() - x.eq(0).cumsum().where(x>0).ffill().fillna(0))
            )
            print(f"   ✓ days_since_last_sale added with {df['days_since_last_sale'].isna().sum()} NaN values")

        # Fill NaN values using interpolation and backfill to ensure data quality
        print("🧹 Handling missing values...")
        print(f"   NaN values before handling: {df.isna().sum().sum()}")

        # Interpolate missing values within each store-item group
        df = df.groupby(['store_nbr', 'item_nbr']).apply(
            lambda g: g.interpolate(method='linear', limit_direction='forward')
        ).reset_index(drop=True)

        # Fill any remaining NaN values with backfill and then with 0
        df = df.fillna(method='bfill').fillna(0)
        print(f"   NaN values after handling: {df.isna().sum().sum()}")

        # Update the class attribute with the engineered features
        self.df = df
        print(f"🎉 Feature engineering complete!")  # noqa: F541
        print(f"   Final data shape: {self.df.shape}")
        print(f"   Final columns: {list(self.df.columns)}")



    # -----------------------------
    # Train/test split
    # -----------------------------
    def split_data(self, target: str = "unit_sales", drop_cols: list = ['store_nbr', 'item_nbr', 'date', 'id']) -> Tuple:
        features =  [
                "onpromotion",
                "onpromotion_lag_1",
                "unit_sales_lag_7",
                "unit_sales_lag_14",
                "unit_sales_lag_30",
                "unit_sales_lag_365",
                "unit_sales_r7_mean",
                "unit_sales_r14_mean",
                "unit_sales_r30_mean",
                "unit_sales_r7_std",
                "day_of_week",
                "month",
                "days_until_holiday",
                "days_since_holiday",
                "store_avg_sales"
            ]
        print("Splitting data into train/test sets...")
        df = self.df
        X = df.drop(columns=[target] + drop_cols)[features]
        y = df[target]
        groups = df['cv_group']

        # Chronological split
        cutoff_date = df['date'].quantile(0.8)
        print(f"Cutoff date for train/test split: {cutoff_date}")
        train_idx = df['date'] <= cutoff_date
        test_idx = df['date'] > cutoff_date

        print(f"Train set size: {train_idx.sum()}, Test set size: {test_idx.sum()}")
        return X.loc[train_idx], X.loc[test_idx], y.loc[train_idx], y.loc[test_idx], groups.loc[train_idx]

    # -----------------------------
    # Preprocessor builder
    # -----------------------------
    def build_preprocessor(self, X: pd.DataFrame, model_type: str) -> ColumnTransformer:
        print(f"Building preprocessor for model type: {model_type}...")
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        object_cols = X.select_dtypes(include=['object']).columns.tolist()
        print(f"Numeric columns: {numeric_cols}")
        print(f"Object columns: {object_cols}")

        if model_type in ["SVR", "Ridge", "LinearRegression"]:
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
        elif model_type in ["RandomForestRegressor", "XGBRegressor"]:
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='median'))
            ])
        else:
            # For LSTM we skip ColumnTransformer, scaling is done in sequence prep
            print("Skipping ColumnTransformer for LSTM model type.")
            return None

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, object_cols)
        ])
        print("Preprocessor built successfully.")
        return preprocessor

    # -----------------------------
    # Top-N feature selection
    # -----------------------------
    def select_top_features(self, X: pd.DataFrame, y: pd.Series, model_type: str, n: int = 20) -> pd.DataFrame:
        print(f"Selecting top {n} features for model type: {model_type}...")
        if model_type in ["SVR", "Ridge", "LinearRegression"]:
            selector = SelectKBest(score_func=f_regression, k=n)
            selector.fit(X, y)
            selected_cols = X.columns[selector.get_support()]
            print(f"Selected columns: {selected_cols.tolist()}")
            return X[selected_cols]
        print("No feature selection for tree-based or LSTM models.")
        return X

    # -----------------------------
    # CV splitter
    # -----------------------------
    def get_cv_splitter(self, n_splits=5):
        print(f"Creating GroupKFold splitter with {n_splits} splits...")
        return GroupKFold(n_splits=n_splits)

    # -----------------------------
    # Convenience: prepare for model
    # -----------------------------
    def prepare_for_model(
        self,
        model_type: str,
        target: str = "unit_sales",
        drop_cols: list = ['store_nbr', 'item_nbr', 'date', 'id',],
        topn: int = 15  # Default to top 15 features
    ):
        print(f"Preparing data for {model_type} model...")
        self.engineer_features()
        X_train, X_test, y_train, y_test, groups_train = self.split_data(target=target, drop_cols=drop_cols)
        
         # Ensure only numeric columns are used for feature selection
        X_train = X_train.select_dtypes(include=['int64', 'float64'])
        X_test = X_test[X_train.columns]
        
        # Ensure no NaN values
        X_train = X_train.interpolate().bfill().ffill()
        X_test = X_test.interpolate().bfill().ffill()


        # Feature selection for linear models only
        if model_type in ["SVR", "LinearRegression"]:
            X_train = self.select_top_features(X_train, y_train, model_type, n=topn)
            X_test = X_test[X_train.columns]  # Align test features with selected features

        # No feature selection for tree-based or LSTM models
        elif model_type in ["RandomForestRegressor", "XGBRegressor", "LSTM"]:
            pass  # Use all features

        preprocessor = self.build_preprocessor(X_train, model_type)
        print("Data preparation for model complete.")
        return X_train, X_test, y_train, y_test, groups_train, preprocessor


    # -----------------------------
    # Prepare for LSTM
    # -----------------------------
    def prepare_for_lstm(self, target: str = "unit_sales", seq_len: int = 30):
        print(f"Preparing data for LSTM with sequence length: {seq_len}...")
        values = self.df[target].values.reshape(-1, 1)

        # Chronological split (80/20)
        train_cut = int(len(values) * 0.8)
        train_raw = values[:train_cut]
        test_raw = values[train_cut:]

        # Fit scaler only on training slice
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_raw)
        test_scaled = scaler.transform(test_raw)

        def make_sequences(arr, seq_len):
            X, y = [], []
            for i in range(len(arr) - seq_len):
                X.append(arr[i:i+seq_len, :])
                y.append(arr[i+seq_len, 0])
            return np.array(X), np.array(y)

        X_train, y_train = make_sequences(train_scaled, seq_len)
        X_test, y_test = make_sequences(test_scaled, seq_len)

        # Returns NumPy arrays shaped exactly as an LSTM expects:
        # X_train/test → (samples, time_steps, features) and y_train/test → (samples,).
        print("Shapes  →  X_train:", X_train.shape,
              "   y_train:", y_train.shape,
              "   X_test:",  X_test.shape,
              "   y_test:",  y_test.shape)

        return X_train, y_train, X_test, y_test, scaler


