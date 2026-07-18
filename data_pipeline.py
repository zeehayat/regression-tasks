# data_pipeline.py
"""
Data preprocessing pipeline to clean and sanitize MHP data.
"""
import pandas as pd
import numpy as np

def sanitize_mhp_data(df, target_col):
    """
    Cleans raw dataframe, drops target leakage, and alerts on missingness.
    """
    cleaned_df = df.copy()
    
    # Identify target leakage (audit codes generated after completion)
    leakage_cols = [col for col in df.columns if 'audit' in col or 'final' in col]
    if leakage_cols:
        print(f"Dropping Target Leakage columns: {leakage_cols}")
        cleaned_df = cleaned_df.drop(columns=leakage_cols)
        
    # Check for missingness
    missing_pct = cleaned_df.isnull().mean()
    for col, pct in missing_pct.items():
        if pct > 0:
            print(f"Column '{col}' is missing {pct*100:.2f}% of values.")
            
    # Safely drop rows with missing targets
    if target_col in cleaned_df.columns:
        cleaned_df = cleaned_df.dropna(subset=[target_col])
        
    return cleaned_df
