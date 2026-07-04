import pandas as pd

# 1. Load your newly clustered dataset
print("Loading clustered dataset...")
df = pd.read_csv("dataset_with_clusters.csv")

# 2. Group the data by Cluster_ID
print("Generating summary report...")
summary = df.groupby('Cluster_ID').agg(
    # Count how many total rows are in this cluster
    Total_Rows=('title', 'count'),
    # Grab the top 3 most common job titles in this cluster as examples
    Example_Titles=('title', lambda x: ' | '.join(x.value_counts().head(3).index))
).reset_index()

# 3. Sort from largest cluster to smallest
summary = summary.sort_values(by='Total_Rows', ascending=False)

# 4. Save to a new CSV for you to review
summary.to_csv("cluster_summary_report.csv", index=False)
print("Saved to cluster_summary_report.csv. Open this file to review!")