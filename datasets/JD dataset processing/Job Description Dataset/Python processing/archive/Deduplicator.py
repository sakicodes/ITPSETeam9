import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan 
import time

def cluster_job_titles(input_csv_path, output_csv_path, title_column='Job Title'):
    """
    Reads a dataset, extracts unique job titles to save compute, clusters them 
    based on semantic meaning, and maps the clusters back to the full dataset.
    """
    print(f"Loading dataset from {input_csv_path}...")
    
    # 1. Load the Full Dataset
    # We load the entire dataset. Ensure 'title_column' matches your actual CSV header.
    df = pd.read_csv(input_csv_path)

    # Store the original row count so you can see how many duplicates get nuked
    original_count = len(df)
    
    # NEW STEP: Drop true row duplicates based on the Job Description text.
    # Replace 'Job Description' with the actual name of your description column!
    df = df.drop_duplicates(subset=['description'], keep='first')
    
    # (Optional) The string cleaning we discussed earlier
    df[title_column] = df[title_column].str.lower().str.strip()

    print(f"Dropped {original_count - len(df)} exact duplicate descriptions.")
    print(f"Dataset reduced from {original_count} to {len(df)} rows.")
    
    if title_column not in df.columns:
        raise ValueError(f"Column '{title_column}' not found in the dataset.")

    # 2. Extract Unique Titles (The "Compute Saver" Step)
    # If you have 123,850 rows, embedding them all takes a long time. 
    # By grabbing only the unique titles, we might drop the workload down to ~10,000 items.
    unique_titles = df[title_column].dropna().unique()
    print(f"Found {len(unique_titles)} unique job titles out of {len(df)} total rows.")
    
    # Convert to a DataFrame for easy mapping later
    unique_df = pd.DataFrame({title_column: unique_titles})

    # 3. Generate Semantic Embeddings
    # We use a lightweight, pre-trained Hugging Face model. 
    # 'all-MiniLM-L6-v2' is excellent for general semantic similarity and runs well on a CPU.
    print("Loading SentenceTransformer model and generating embeddings...")
    start_time = time.time()
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # This converts our text titles into mathematical vectors (lists of numbers).
    # Titles with similar meanings (e.g., "UI Dev" and "Front End Developer") 
    # will have mathematical vectors that are close to each other.
    embeddings = model.encode(unique_df[title_column].tolist(), show_progress_bar=True)
    
    print(f"Embeddings generated in {time.time() - start_time:.2f} seconds.")

    # 4. Dimensionality Reduction (UMAP)
    # The embeddings we just generated have 384 dimensions. HDBSCAN struggles to cluster 
    # efficiently in highly multi-dimensional space (the "curse of dimensionality").
    # UMAP compresses these 384 dimensions down to 5, preserving the local structure/meaning.
    print("Reducing dimensionality with UMAP...")
    umap_model = umap.UMAP(
        n_neighbors=15,    # Looks at 15 neighboring titles to build the local structure
        n_components=5,    # Reduces down to 5 dimensions for clustering
        metric='cosine',   # Cosine similarity is best for text embeddings
        random_state=42    # Sets a seed so your results are reproducible
    )
    reduced_embeddings = umap_model.fit_transform(embeddings)

    # 5. Density-Based Clustering (HDBSCAN)
    # HDBSCAN groups the compressed vectors based on density. 
    # It is brilliant for this use case because it doesn't force everything into a cluster.
    # Truly random noise (e.g., "Chief Happiness Officer") will be given a cluster ID of -1.
    print("Clustering titles with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,     # A cluster must have at least 5 similar titles to be formed
        min_samples=2,          # How conservative the clustering is (lower = more clusters)
        cluster_selection_method='eom' # "Excess of Mass" - standard approach for HDBSCAN
    )
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    
    # Add the cluster labels back to our unique titles dataframe
    unique_df['Cluster_ID'] = cluster_labels

    # 6. Map Clusters Back to the Main Dataset
    # Now we take our cleanly clustered unique titles and merge them back into the massive 123k dataset.
    print("Merging clusters back to the main dataset...")
    final_df = pd.merge(df, unique_df, on=title_column, how='left')

    # 7. Save the Output
    final_df.to_csv(output_csv_path, index=False)
    
    # Print a quick summary of the results
    total_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    noise_count = list(cluster_labels).count(-1)
    
    print("\n--- Clustering Complete ---")
    print(f"Data saved to: {output_csv_path}")
    print(f"Total valid clusters found: {total_clusters}")
    print(f"Number of 'noise' titles (Cluster ID -1): {noise_count}")
    
    return final_df

# ==========================================
# HOW TO RUN THE SCRIPT
# ==========================================
if __name__ == "__main__":
    # Replace these strings with your actual file names/paths
    INPUT_FILE = "postings.csv"    
    OUTPUT_FILE = "dataset_with_clusters.csv"
    
    # Replace 'COLUMN_NAME' with whatever the actual column header is in your CSV
    COLUMN_NAME = "title" 
    
    # Uncomment the line below to execute when you have your files ready:
    clustered_data = cluster_job_titles(INPUT_FILE, OUTPUT_FILE, title_column=COLUMN_NAME)