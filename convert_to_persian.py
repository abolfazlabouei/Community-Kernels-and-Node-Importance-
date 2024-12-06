import pandas as pd
import numpy as np
import os
from finglish import f2p
from concurrent.futures import ThreadPoolExecutor, as_completed

# Display number of CPU cores
num_cores = os.cpu_count()
print(f"Number of CPU cores available: {num_cores}")

# Load your DataFrame
df = pd.read_csv("more_200_drop_type&participants.csv")

# Output file path
output_path = "converted_to_persian.csv"

# Initialize the output file with headers
df.iloc[:0].to_csv(output_path, index=False)

# Optimized function for converting Finglish to Persian
def convert_to_persian_chunk(chunk, chunk_id, columns):
    result = chunk.copy()
    for col in columns:
        processed_col = []
        for i, text in enumerate(chunk[col]):
            if pd.notnull(text):
                try:
                    processed_col.append(f2p(text))
                except Exception as e:
                    print(f"Error converting text in chunk {chunk_id}, Column {col}, index {i}: {text}, Error: {e}")
                    processed_col.append(text)
            else:
                processed_col.append(text)
            # Log progress every 100 rows
            if (i + 1) % 5000 == 0:
                print(f"Chunk {chunk_id}, Column {col}: Processed {i + 1}/{len(chunk[col])} rows")
        result[col] = processed_col
    return result

# Process chunks sequentially
def process_chunks_sequentially(df, columns, chunk_size):
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)
    for chunk_id in range(num_chunks):
        # Extract the chunk
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx]

        # Process the chunk
        processed_chunk = convert_to_persian_chunk(chunk, chunk_id, columns)

        # Save the processed chunk to the file
        processed_chunk.to_csv(output_path, mode='a', header=False, index=False)
        print(f"Chunk {chunk_id} processed and saved ({start_idx} to {end_idx})")

# Columns to process
columns_to_convert = ['peer_name', 'about']

# Convert columns in chunks sequentially
chunk_size = 1000  # Adjust the chunk size as needed
process_chunks_sequentially(df, columns_to_convert, chunk_size)

print(f"Real-time conversion completed and saved to {output_path}")
