"""
Feature Orchestration Layer for Loan Default Prediction

This module combines individual feature engineering functions from 
feature_engineering.py to create complete feature sets for model training.

Main Functions:
1. traditional_features() - Combines ALL Home Credit datasets (487 features)
   Includes: apps + prev + bureau + pos + installments + credit cards
   
2. behavioral_features() - Extracts UCI Credit Card behavioral features (31 features)
   Includes: payment patterns, spending volatility, financial stress indicators

Date: November 2025
Version: 3.0
"""

import pandas as pd
import logging
from .feature_engineering import (
    process_apps,
    process_prev,
    get_prev_agg,
    process_bureau,
    get_bureau_agg,
    process_pos,
    process_install,
    process_card,
    behaviorial_features,
)

logger = logging.getLogger(__name__)


def traditional_features(apps, bureau, bureau_bal, prev, pos_bal, install, card_bal):
    """
    Create traditional feature set combining ALL data sources.rces.
    
    This is the most comprehensive feature set, combining traditional credit
    data with behavioral patterns from balance histories.
    
    Parameters:
    -----------
    apps : pd.DataFrame
        Application data
    bureau : pd.DataFrame
        Credit bureau data
    bureau_bal : pd.DataFrame
        Bureau balances
    prev : pd.DataFrame
        Previous applications
    pos_bal : pd.DataFrame
        POS cash balance history
    install : pd.DataFrame
        Installments payments
    card_bal : pd.DataFrame
        Credit card balance history
    
    Returns:
    --------
    pd.DataFrame
        Complete dataset with 487 features including:
        - Application features (13)
        - Previous loan aggregations (PREV_*)
        - Bureau aggregations (BUREAU_*)
        - POS balance features (POS_*)
        - Installment features (INSTALL_*)
        - Credit card features (CARD_*)
    
    Pipeline:
    ---------
    1. Process applications
    2. Aggregate each data source
    3. Sequential left joins on SK_ID_CURR
    4. Memory optimization
    
    Used By:
    --------
    - Traditional Home Credit model (Traditional_model.pkl)
    - Ensemble model's traditional branch
    
    Example:
    --------
    >>> apps, prev, bureau, bb = get_dataset()
    >>> pos, install, card = get_balance_data()
    >>> traditional_df = traditional_features(apps, bureau, bb, prev, pos, install, card)
    >>> print(f"Total features: {traditional_df.shape[1]}")
    Total features: 487
    """

    apps_all =  process_apps(apps)
    bureau_agg = get_bureau_agg(bureau, bureau_bal)
    prev_agg = get_prev_agg(prev)
    pos_bal_agg = process_pos(pos_bal)
    install_agg = process_install(install)
    card_bal_agg = process_card(card_bal)
    # logger.debug('prev_agg shape: %s bureau_agg shape: %s', prev_agg.shape, bureau_agg.shape)
    logger.info('pos_bal_agg shape: %s install_agg shape: %s card_bal_agg shape: %s', pos_bal_agg.shape, install_agg.shape, card_bal_agg.shape)
    logger.info('apps_all before merge shape: %s', apps_all.shape)

    # Join with apps_all
    apps_all = apps_all.merge(prev_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(bureau_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(pos_bal_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(install_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(card_bal_agg, on='SK_ID_CURR', how='left')

    logger.info('apps_all after merge with all shape: %s', apps_all.shape)
    # apps_all = reduce_mem_usage(apps_all) # Apply memory reduction after merging
    print('data types are converted for a reduced memory usage')


    return apps_all


def behavioral_features(uci_data):
    """
    Extract behavioral features from UCI Credit Card dataset.
    
    Uses the behaviorial_features() function from feature_engineering.py
    to create 31 behavioral features analyzing payment patterns and
    financial behavior from credit card history.
    
    Parameters:
    -----------
    uci_data : pd.DataFrame
        UCI Credit Card dataset with:
        - LIMIT_BAL: Credit limit
        - SEX, EDUCATION, MARRIAGE, AGE: Demographics
        - PAY_0 to PAY_6: Payment status history
        - BILL_AMT1 to BILL_AMT6: Monthly bill amounts
        - PAY_AMT1 to PAY_AMT6: Monthly payment amounts
    
    Returns:
    --------
    pd.DataFrame
        Original data + 31 engineered behavioral features including:
        - Aggregate features (total bills, payments, averages)
        - Volatility indicators (spending & payment consistency)
        - Financial stress indicators (debt stress, missed payments)
        - Behavioral ratios (payment consistency, utilization trends)
    
    Pipeline:
    ---------
    1. Loads UCI dataset
    2. Applies behaviorial_features() from feature_engineering.py
    3. Returns enriched dataset with behavioral patterns
    
    Used By:
    --------
    - Behavioral LightGBM model (Behaviorial_model.pkl)
    - Ensemble model's behavioral branch
    
    Example:
    --------
    >>> uci_df = pd.read_csv('data/uci_credit_card.csv')
    >>> uci_features = behavioral_features(uci_df)
    >>> print(f"Behavioral features: {uci_features.shape[1]}")
    Behavioral features: 31
    """
    return behaviorial_features(uci_data)


def get_apps_all_encoded(apps_all):
    """
    Encode categorical variables using factorization.
    
    Converts all object-type columns to numeric codes for model training.
    
    Parameters:
    -----------
    apps_all : pd.DataFrame
        Feature-engineered dataset with mixed types
    
    Returns:
    --------
    pd.DataFrame
        Same dataset with object columns converted to numeric codes
    
    Note:
    -----
    Uses pd.factorize() which creates integer codes for each unique value.
    Better for tree-based models than one-hot encoding.
    
    Example:
    --------
    >>> df_encoded = get_apps_all_encoded(df_with_categories)
    >>> print(df_encoded.dtypes.value_counts())
    """

    object_columns = apps_all.dtypes[apps_all.dtypes == 'object'].index.tolist()
    for column in object_columns:
        apps_all[column] = pd.factorize(apps_all[column])[0]

    return apps_all
