import fasttext
import pandas as pd
import logging
from collections import Counter
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Load the pre-trained FastText model
logging.info("Loading FastText model...")
model = fasttext.load_model("cc.fa.300.bin")
logging.info("Model loaded successfully.")

# Function to get most frequent words in the dataset
def extract_frequent_words(file, column, top_n=50):
    logging.info("Extracting frequent words from the dataset...")
    word_counter = Counter()
    for chunk in pd.read_csv(file, chunksize=1000):
        for text in chunk[column].dropna():
            word_counter.update(text.split())
    frequent_words = [word for word, _ in word_counter.most_common(top_n)]
    logging.info(f"Frequent words extracted: {frequent_words}")
    return frequent_words

# Function to calculate average vector for a text
def calculate_average_vector(text, model):
    words = text.split()
    vectors = [model[word] for word in words if word in model]
    if not vectors:  # Handle cases where no word is in the model
        return np.zeros(model.get_dimension())
    return np.mean(vectors, axis=0)

# Function to assign labels automatically
def assign_label_auto(text, model, labels):
    if not text:  # Handle empty or NaN text
        return "Other"
    text_vector = calculate_average_vector(text, model)
    similarities = {
        label: np.dot(text_vector, model[label]) for label in labels if label in model
    }
    if not similarities:
        return "Other"
    return max(similarities, key=similarities.get)  # Return the label with the highest similarity

# Input and output file paths
input_file = "clean_df.csv"
output_file = "labels_output.csv"  # File to store only labels

# Extract frequent words from the dataset
frequent_labels = extract_frequent_words(input_file, column="combined_cleaned_text", top_n=50)

# Function to process file in chunks using yield
def process_in_chunks(input_file, chunk_size=100):
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        chunk['combined_cleaned_text'] = chunk['combined_cleaned_text'].fillna("")
        chunk['label'] = chunk['combined_cleaned_text'].apply(lambda x: assign_label_auto(x, model, frequent_labels))
        yield chunk['label']  # Yield only the label column

# Open the output file
header_written = False

logging.info("Starting to process the file in chunks...")

# Iterate over chunks
for chunk_index, labels_chunk in enumerate(process_in_chunks(input_file, chunk_size=100)):
    logging.info(f"Processing chunk {chunk_index + 1}...")

    # Save only labels to the output file
    labels_chunk.to_csv(output_file, mode='a', index=False, header=not header_written, encoding='utf-8-sig')
    header_written = True

    logging.info(f"Chunk {chunk_index + 1} processed: {len(labels_chunk)} rows.")

logging.info("File processing completed.")


