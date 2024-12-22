import pandas as pd
import os

def csv_to_fasttext_format(folder_path, output_file):
    """
    Reads CSV files from a folder, extracts the 'peer_name' column as text, 
    and saves them in FastText format using the file name as the label.

    Args:
        folder_path (str): Path to the folder containing CSV files.
        output_file (str): Path to the output .txt file for FastText.
    """
    print(f"Processing folder: {folder_path}")
    print(f"Files in folder: {os.listdir(folder_path)}")

    # Print absolute path of the output file to ensure it's being saved in the correct location
    print(f"Output will be saved to: {os.path.abspath(output_file)}")

    with open(output_file, "w", encoding="utf-8") as f_out:
        # Iterate over all files in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):  # Check if it's a CSV file
                file_path = os.path.join(folder_path, file_name)
                label = os.path.splitext(file_name)[0]  # Use file name (without extension) as the label
                print(f"\nProcessing file: {file_name}")
                
                # Read CSV file
                try:
                    df = pd.read_csv(file_path)
                except Exception as e:
                    print(f"Error reading {file_name}: {e}")
                    continue

                # Debug columns
                print(f"Columns in {file_name}: {df.columns}")

                # Check if 'peer_name' column exists
                if 'peer_name' not in df.columns:
                    print(f"'peer_name' column not found in {file_name}")
                    continue

                # Check non-empty rows
                non_empty_rows = df['peer_name'].dropna()
                print(f"Non-empty rows in 'peer_name' for {file_name}: {non_empty_rows.shape[0]}")

                # Extract 'peer_name' column and save in FastText format
                if non_empty_rows.empty:
                    print(f"No valid 'peer_name' entries found in {file_name}")
                for text in non_empty_rows:
                    f_out.write(f"__label__{label} {text}\n")

    print(f"\nData successfully written to {os.path.abspath(output_file)}")

# Example usage
folder_path = "folder path"  # Replace with the path to your folder containing CSV files
output_file = "train_data.txt"  # Replace with your desired output file path
csv_to_fasttext_format(folder_path, output_file)

