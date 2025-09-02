# ===================================================================
# CELL 1: All Imports and Functions
# ===================================================================
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import xgboost as xgb

# Part 1: Corrected Data Exploration
def explore_data(root_path):
    """
    Loads key data files and performs basic EDA with corrected parameters.
    """
    print(f"--- Starting Data Exploration in: {root_path} ---\n")
    file_configs = {
        "cgm": {"path": os.path.join(root_path, "cgm.txt"), "sep": '|', "encoding": 'utf-8'},
        "bolus": {"path": os.path.join(root_path, "Pump_BolusDelivered.txt"), "sep": '|', "encoding": 'utf-8'},
        "roster": {"path": os.path.join(root_path, "PtRoster_a.txt"), "sep": '\t', "encoding": 'utf-16'},
        "screening": {"path": os.path.join(root_path, "DiabScreening_a.txt"), "sep": '\t', "encoding": 'utf-16'},
    }

    for name, config in file_configs.items():
        path = config["path"]
        print(f"--- Exploring {name} ({os.path.basename(path)}) ---")
        if not os.path.exists(path):
            print(f"Warning: File not found at {path}\n")
            continue
        try:
            df = pd.read_csv(path, sep=config["sep"], encoding=config["encoding"], low_memory=False, on_bad_lines='skip')
            print("\n[INFO] First 5 rows:")
            print(df.head())
            print("\n[INFO] Dataframe Info:")
            df.info()
            print("-" * 50 + "\n")
        except Exception as e:
            print(f"Could not process file {path}. Error: {e}\n")

# Part 2: Data Preprocessing
def run_preprocessing(cgm_filepath, bolus_filepath, output_filepath):
    """
    Loads raw data, performs all cleaning and feature engineering,
    and saves the final dataset to a Parquet file.
    """
    print("--- Starting Data Preprocessing ---")
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

    print("\nStep 2: Engineering features...")
    final_df = final_df.sort_values(by=['PtID', 'DataDtTm']).reset_index(drop=True)
    prediction_horizon = 6
    final_df['glucose_target'] = final_df.groupby('PtID')['CGM'].shift(-prediction_horizon)
    for lag in [1, 3, 6, 12]:
        final_df[f'glucose_lag_{lag}'] = final_df.groupby('PtID')['CGM'].shift(lag)
    for window in [6, 12, 24]:
        final_df[f'glucose_rolling_mean_{window}'] = final_df.groupby('PtID')['CGM'].transform(lambda x: x.rolling(window, 1).mean())
        final_df[f'glucose_rolling_std_{window}'] = final_df.groupby('PtID')['CGM'].transform(lambda x: x.rolling(window, 1).std())
    final_df['BolusAmount'] = final_df['BolusAmount'].fillna(0)
    final_df.dropna(subset=['glucose_target', 'glucose_lag_12', 'CGM'], inplace=True)
    final_df = final_df[np.isfinite(final_df['glucose_rolling_std_24'])].reset_index(drop=True)
    print(f"Feature engineering complete. Final dataset has {len(final_df)} rows.")

    print(f"\nStep 3: Saving processed data to {output_filepath}...")
    final_df.to_parquet(output_filepath, index=False)
    print("--- Preprocessing Finished Successfully! ---")

# Part 3: Model Training with Optuna and XGBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix

def objective_xgb(trial, X_train_cpu, y_train_cpu, X_val_cpu, y_val_cpu, weights):
    """
    Trains an XGBoost model with hyperparameters suggested by Optuna.
    """
    params = {
        'device': 'cuda',  # Use modern syntax for GPU training
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 400, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'random_state': 42,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train_cpu, y_train_cpu, sample_weight=weights)
    y_pred = model.predict(X_val_cpu)
    rmse = np.sqrt(mean_squared_error(y_val_cpu, y_pred))
    return rmse

def run_xgb_tuning(processed_filepath):
    """
    Loads data, runs Optuna with XGBoost, and trains a final model.
    """
    print("\n--- Starting Hyperparameter Tuning with Optuna & XGBoost ---")
    print(f"Step 1: Loading data from {processed_filepath}...")
    final_df = pd.read_parquet(processed_filepath)
    features = [col for col in final_df.columns if 'glucose_lag' in col or 'glucose_rolling' in col or 'BolusAmount' in col]
    target = 'glucose_target'
    X_cpu = final_df[features].astype(np.float32)
    y_cpu = final_df[target].astype(np.float32)

    X_train_full_cpu, X_test_cpu, y_train_full_cpu, y_test_cpu = train_test_split(
        X_cpu, y_cpu, test_size=0.2, random_state=42, stratify=final_df['PtID']
    )
    X_train_cpu, X_val_cpu, y_train_cpu, y_val_cpu = train_test_split(
        X_train_full_cpu, y_train_full_cpu, test_size=0.25, random_state=42
    )
    print(f"Data split: {len(X_train_cpu)} train, {len(X_val_cpu)} validation, {len(X_test_cpu)} test samples.")

    print("Creating sample weights to address class imbalance...")
    weights = np.ones(len(y_train_cpu))
    weights[y_train_cpu < 70] = 20
    weights[y_train_cpu > 180] = 1.5

    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective_xgb(trial, X_train_cpu, y_train_cpu, X_val_cpu, y_val_cpu, weights),
        n_trials=30
    )

    print("\n--- Optuna Study Finished ---")
    print(f"Best validation RMSE: {study.best_value}")
    print("Best hyperparameters found:")
    print(study.best_params)

    print("\nStep 3: Training final XGBoost model...")
    final_params = study.best_params
    final_params['device'] = 'cuda'
    final_model = xgb.XGBRegressor(**final_params, random_state=42)
    
    full_weights = np.ones(len(y_train_full_cpu))
    full_weights[y_train_full_cpu < 70] = 20
    full_weights[y_train_full_cpu > 180] = 1.5
    final_model.fit(X_train_full_cpu, y_train_full_cpu, sample_weight=full_weights)
    print("Final model training complete.")

    print("\nStep 4: Evaluating the final model on the test set...")
    y_pred_final = final_model.predict(X_test_cpu)
    
    print("\n--- Final Model Performance on Test Set ---")
    y_test = y_test_cpu.to_numpy()
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
    r2 = r2_score(y_test, y_pred_final)
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} mg/dL")
    print(f"R-squared (R²): {r2:.2f}")

    print("\n--- Classification Metrics ---")
    def to_glycemic_category(glucose_values):
        bins = [0, 70, 180, 500]
        labels = ['Low', 'Normal', 'High']
        return pd.cut(glucose_values, bins=bins, labels=labels, right=False)

    y_test_cat = to_glycemic_category(y_test)
    y_pred_cat = to_glycemic_category(y_pred_final)
    
    y_test_cat = y_test_cat.astype(str)
    y_pred_cat = pd.Series(y_pred_cat).astype(str)
    y_pred_cat[y_pred_cat == 'nan'] = 'Normal'

    print(classification_report(y_test_cat, y_pred_cat))

    print("\nGenerating confusion matrix...")
    cm_labels = ['Low', 'Normal', 'High']
    cm = confusion_matrix(y_test_cat, y_pred_cat, labels=cm_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels, yticklabels=cm_labels)
    plt.title('Confusion Matrix for Glycemic Status Prediction')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# ===================================================================
# CELL 2: Define File Paths and Run the Full Pipeline
# ===================================================================

# --- USER ACTION REQUIRED: Define file paths ---
data_root_path = "/content/drive/MyDrive/A glucose monitor/DCLP3/Data Files/"
output_dir = "/content/drive/MyDrive/A glucose monitor/output/"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

cgm_file = os.path.join(data_root_path, "cgm.txt")
bolus_file = os.path.join(data_root_path, "Pump_BolusDelivered.txt")
processed_output_file = os.path.join(output_dir, "processed_glucose_data.parquet")

# --- Run the full pipeline ---

# 1. Explore the raw data
explore_data(data_root_path)

# 2. Preprocess the data and save the result
run_preprocessing(cgm_filepath=cgm_file, bolus_filepath=bolus_file, output_filepath=processed_output_file)

# 3. Run hyperparameter tuning and train the final model
run_xgb_tuning(processed_filepath=processed_output_file)
