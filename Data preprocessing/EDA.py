import os
import pandas as pd

def explore_data(root_path):
    """
    Loads key data files from the DCLP3 dataset and performs basic EDA.

    Args:
        root_path (str): The path to the 'Data Files' directory.
    """
    print(f"Starting EDA for data in: {root_path}\n")

    # Define paths to the most critical files for prediction
    # Note: The file names are based on the provided repository structure.
    file_paths = {
        "cgm": os.path.join(root_path, "cgm.txt"),
        "bolus": os.path.join(root_path, "Pump_BolusDelivered.txt"),
        "roster": os.path.join(root_path, "PtRoster_a.txt"),
        "screening": os.path.join(root_path, "DiabScreening_a.txt"),
    }

    # Load and explore each file
    for name, path in file_paths.items():
        print(f"--- Exploring {name} ({os.path.basename(path)}) ---")
        if not os.path.exists(path):
            print(f"Warning: File not found at {path}\n")
            continue

        try:
            # Load the data. Assuming tab-separated text files.
            # You may need to adjust the separator, e.g., sep=',' for CSV.
            df = pd.read_csv(path, sep='\t', low_memory=False)

            print("\n[INFO] First 5 rows:")
            print(df.head())

            print("\n[INFO] Dataframe Info (columns, data types, non-null counts):")
            df.info()

            print("\n[INFO] Descriptive Statistics for numeric columns:")
            print(df.describe())

            print("-" * 50 + "\n")

        except Exception as e:
            print(f"Could not process file {path}. Error: {e}\n")

# --- USER ACTION REQUIRED ---
# Set the root path to your "Data Files" directory
# This path should point to the folder containing cgm.txt, PtRoster_a.txt, etc.
your_data_path = "/content/drive/MyDrive/A glucose monitor/DCLP3/Data Files/"

# Run the exploration function
explore_data(your_data_path)
