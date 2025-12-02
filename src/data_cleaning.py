"""
Data Cleaning Module
Centralized data cleaning and missing value handling for the Loan Default Hybrid System.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop ID columns from dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with ID columns removed
    """
    id_columns = ['ID', 'id', 'Id', 'SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU']
    columns_to_drop = [col for col in id_columns if col in df.columns]
    
    if columns_to_drop:
        df = df.drop(columns_to_drop, axis=1)
        print(f"Dropped ID columns: {columns_to_drop}")
    
    return df


def clean_target_column(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Remove rows with missing values in the target column.
    
    Args:
        df: Input dataframe
        target_column: Name of the target column
        
    Returns:
        Dataframe with NaN target values removed
    """
    initial_rows = len(df)
    df = df.dropna(subset=[target_column])
    removed_rows = initial_rows - len(df)
    
    if removed_rows > 0:
        print(f"Removed {removed_rows} rows with missing target values")
    
    return df


def replace_placeholder_values(df: pd.DataFrame, placeholder: Union[int, float] = 365243) -> pd.DataFrame:
    """
    Replace placeholder values with NaN.
    Common placeholder value in credit data is 365243 (representing missing data).
    
    Args:
        df: Input dataframe
        placeholder: Placeholder value to replace
        
    Returns:
        Dataframe with placeholders replaced by NaN
    """
    df = df.replace(placeholder, np.nan)
    return df


def handle_infinities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace infinite values with NaN.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with infinities replaced by NaN
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def impute_categorical_columns(df: pd.DataFrame, 
                               categorical_columns: Optional[List[str]] = None,
                               fill_value: str = 'MISSING') -> pd.DataFrame:
    """
    Fill missing values in categorical columns with a default value.
    
    Args:
        df: Input dataframe
        categorical_columns: List of categorical column names. If None, auto-detect object columns
        fill_value: Value to use for filling missing categorical data
        
    Returns:
        Dataframe with categorical NaNs filled
    """
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna(fill_value)
    
    return df


def impute_numeric_columns(df: pd.DataFrame, 
                          numeric_columns: Optional[List[str]] = None,
                          strategy: str = 'median') -> pd.DataFrame:
    """
    Fill missing values in numeric columns using specified strategy.
    
    Args:
        df: Input dataframe
        numeric_columns: List of numeric column names. If None, auto-detect numeric columns
        strategy: Imputation strategy - 'median', 'mean', or 'zero'
        
    Returns:
        Dataframe with numeric NaNs filled
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_columns:
        if col in df.columns:
            if strategy == 'median':
                fill_value = df[col].median()
                # Fallback to 0 if median is NaN (all values missing)
                if pd.isna(fill_value):
                    fill_value = 0
            elif strategy == 'mean':
                fill_value = df[col].mean()
                if pd.isna(fill_value):
                    fill_value = 0
            elif strategy == 'zero':
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            df[col] = df[col].fillna(fill_value)
    
    return df


def align_features(X: pd.DataFrame, expected_features: List[str], fill_value: float = 0) -> pd.DataFrame:
    """
    Align dataframe features with expected feature list.
    - Adds missing features with default value
    - Removes extra features not in expected list
    - Maintains correct column order
    
    Args:
        X: Input dataframe
        expected_features: List of expected feature names
        fill_value: Default value for missing features
        
    Returns:
        Aligned dataframe with expected features
    """
    # Find missing features
    missing_features = [f for f in expected_features if f not in X.columns]
    
    # Add missing features
    if missing_features:
        missing_dict = {col: fill_value for col in missing_features}
        missing_df = pd.DataFrame(missing_dict, index=X.index)
        X = pd.concat([X, missing_df], axis=1)
    
    # Select and reorder to match expected features
    X = X[expected_features]
    
    return X


def clean_dataframe(df: pd.DataFrame, 
                   target_column: Optional[str] = None,
                   drop_ids: bool = True,
                   handle_placeholders: bool = True,
                   placeholder_value: Union[int, float] = 365243,
                   categorical_fill: str = 'MISSING',
                   numeric_strategy: str = 'median') -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    
    Args:
        df: Input dataframe
        target_column: Name of target column (will remove rows with missing targets)
        drop_ids: Whether to drop ID columns
        handle_placeholders: Whether to replace placeholder values with NaN
        placeholder_value: Placeholder value to replace
        categorical_fill: Fill value for categorical columns
        numeric_strategy: Strategy for numeric imputation ('median', 'mean', 'zero')
        
    Returns:
        Cleaned dataframe
    """
    df = df.copy()
    
    # Step 1: Drop ID columns
    if drop_ids:
        df = drop_id_columns(df)
    
    # Step 2: Clean target column (must be before other cleaning)
    if target_column and target_column in df.columns:
        df = clean_target_column(df, target_column)
    
    # Step 3: Replace placeholder values
    if handle_placeholders:
        df = replace_placeholder_values(df, placeholder_value)
    
    # Step 4: Handle infinities
    df = handle_infinities(df)
    
    # Step 5: Separate features and target
    if target_column:
        y = df[target_column]
        X = df.drop(target_column, axis=1)
    else:
        X = df
        y = None
    
    # Step 6: Impute categorical columns
    X = impute_categorical_columns(X, fill_value=categorical_fill)
    
    # Step 7: Impute numeric columns
    X = impute_numeric_columns(X, strategy=numeric_strategy)
    
    # Step 8: Recombine if we had a target
    if y is not None:
        df = X.copy()
        df[target_column] = y
    else:
        df = X
    
    return df


def prepare_prediction_data(X: pd.DataFrame, 
                           expected_features: List[str],
                           categorical_fill: str = 'MISSING',
                           numeric_strategy: str = 'median') -> pd.DataFrame:
    """
    Prepare data for prediction by cleaning and aligning features.
    This is used during inference when we don't have a target column.
    
    Args:
        X: Input feature dataframe
        expected_features: List of features expected by the model
        categorical_fill: Fill value for categorical columns
        numeric_strategy: Strategy for numeric imputation
        
    Returns:
        Cleaned and aligned dataframe ready for prediction
    """
    X = X.copy()
    
    # Clean the data
    X = replace_placeholder_values(X)
    X = handle_infinities(X)
    X = impute_categorical_columns(X, fill_value=categorical_fill)
    X = impute_numeric_columns(X, strategy=numeric_strategy)
    
    # Align features
    X = align_features(X, expected_features, fill_value=0)
    
    return X


# Backward compatibility: Keep the specific cleaning functions used in feature engineering
def clean_feature_engineering_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Specific cleaning for feature engineering pipeline.
    Handles the exact transformations previously in feature_engineering.py
    
    Args:
        df: Input dataframe
        
    Returns:
        Cleaned dataframe
    """
    df = df.copy()
    
    # Replace placeholder value
    df = df.replace(365243, np.nan)
    
    # Handle infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Specific column imputations from feature_engineering.py
    if 'CODE_GENDER' in df.columns:
        df['CODE_GENDER'] = df['CODE_GENDER'].fillna('XNA')
    
    if 'NAME_EDUCATION_TYPE' in df.columns:
        df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].fillna('Unknown')
    
    if 'NAME_FAMILY_STATUS' in df.columns:
        df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].fillna('Unknown')
    
    if 'AMT_ANNUITY' in df.columns:
        df['AMT_ANNUITY'] = df['AMT_ANNUITY'].fillna(df['AMT_ANNUITY'].median())
    
    if 'AMT_GOODS_PRICE' in df.columns:
        df['AMT_GOODS_PRICE'] = df['AMT_GOODS_PRICE'].fillna(df['AMT_GOODS_PRICE'].median())
    
    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].fillna(df['DAYS_EMPLOYED'].median())
    
    # Fill remaining numeric nulls with median
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isna().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)
    
    # Fill remaining categorical nulls with 'MISSING'
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        if df[col].isna().any():
            df[col] = df[col].fillna('MISSING')
    
    return df
