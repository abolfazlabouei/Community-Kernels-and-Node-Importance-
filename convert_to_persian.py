

import pandas as pd
import numpy as np
from finglish import f2p
from concurrent.futures import ThreadPoolExecutor

# Load the first 50 rows of your DataFrame
df = pd.read_csv("more_200_drop_type&participants.csv", nrows=50)

# Optimized function for converting Finglish to Persian
def convert_to_persian_vectorized(texts):
    return [f2p(text) if pd.notnull(text) else text for text in texts]

# Apply conversion in parallel using ThreadPoolExecutor
def parallel_apply(df, columns, func):
    with ThreadPoolExecutor() as executor:
        results = {
            col: executor.submit(func, df[col].values) for col in columns
        }
        for col, future in results.items():
            df[col] = future.result()
    return df

# Columns to process
columns_to_convert = ['peer_name', 'about']

# Convert columns in parallel
df = parallel_apply(df, columns_to_convert, convert_to_persian_vectorized)

# Save the updated DataFrame
output_path = "converted_to_persian_50_rows.csv"
df.to_csv(output_path, index=False)
print(f"Converted DataFrame (50 rows) saved to {output_path}")
