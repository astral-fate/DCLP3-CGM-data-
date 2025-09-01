# FILE 1: preprocess_data.py

import pandas as pd
import numpy as np
import os

def run_preprocessing(cgm_filepath, bolus_filepath, output_filepath):
    """
    Loads raw data, performs all cleaning and feature engineering,
    and saves the final dataset to a Parquet file.
    """
    print("--- Starting Data Preprocessing ---")

    # --- 1. Load and Merge Data ---
    print("Step 1: Loading and preparing data...")
    try:
        cgm_df = pd.read_csv(cgm_filepath, sep='|', on_bad_lines='skip')
        bolus_df = pd.read_csv(bolus_filepath, sep='|', on_bad_lines='skip')
        cgm_df.columns = cgm_df.columns.str.strip()
        bolus_df.columns = bolus_df.columns.str.strip()
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the file paths are correct.")
        return

    print("Converting timestamps...")
    cgm_df['DataDtTm'] = pd.to_datetime(cgm_df['DataDtTm'], format='%d%b%y:%H:%M:%S', errors='coerce')
    bolus_df['DataDtTm'] = pd.to_datetime(bolus_df['DataDtTm'], errors='coerce')

    cgm_df.dropna(subset=['DataDtTm'], inplace=True)
    bolus_df.dropna(subset=['DataDtTm'], inplace=True)

    cgm_df['CGM'] = pd.to_numeric(cgm_df['CGM'], errors='coerce')
    bolus_df['BolusAmount'] = pd.to_numeric(bolus_df['BolusAmount'], errors='coerce')

    print("Merging dataframes...")
    final_df = pd.merge_asof(
        cgm_df.sort_values('DataDtTm'),
        bolus_df.sort_values('DataDtTm'),
        on='DataDtTm',
        by='PtID',
        direction='nearest',
        tolerance=pd.Timedelta('5 minutes')
    )

    # --- 2. Feature Engineering ---
    print("\nStep 2: Engineering features...")
    final_df = final_df.sort_values(by=['PtID', 'DataDtTm']).reset_index(drop=True)

    prediction_horizon = 6
    final_df['glucose_target'] = final_df.groupby('PtID')['CGM'].shift(-prediction_horizon)

    for lag in [1, 3, 6, 12]:
        final_df[f'glucose_lag_{lag}'] = final_df.groupby('PtID')['CGM'].shift(lag)

    for window in [6, 12, 24]:
        final_df[f'glucose_rolling_mean_{window}'] = final_df.groupby('PtID')['CGM'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        final_df[f'glucose_rolling_std_{window}'] = final_df.groupby('PtID')['CGM'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

    final_df['BolusAmount'] = final_df['BolusAmount'].fillna(0)
    final_df.dropna(subset=['glucose_target', 'glucose_lag_12', 'CGM'], inplace=True)
    final_df = final_df[np.isfinite(final_df['glucose_rolling_std_24'])].reset_index(drop=True)
    print(f"Feature engineering complete. Final dataset has {len(final_df)} rows.")

    # --- 3. Save the Processed Data ---
    print(f"\nStep 3: Saving processed data to {output_filepath}...")
    final_df.to_parquet(output_filepath, index=False)
    print("--- Preprocessing Finished Successfully! ---")


# --- USER ACTION REQUIRED ---
cgm_file = "/content/drive/MyDrive/A glucose monitor/DCLP3/Data Files/cgm.txt"
bolus_file = "/content/drive/MyDrive/A glucose monitor/DCLP3/Data Files/Pump_BolusDelivered.txt"



processed_output_file = "/content/drive/MyDrive/A glucose monitor/output/processed_glucose_data.parquet"

# Run the preprocessing pipeline
run_preprocessing(cgm_filepath=cgm_file, bolus_filepath=bolus_file, output_filepath=processed_output_file)
