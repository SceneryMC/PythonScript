import pandas as pd
import numpy as np
import matplotlib

# Set non-interactive backend for server use
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def analyze_and_visualize_geometry(csv_filename: str):
    """
    Performs a detailed analysis of the geometry within IVF clusters
    based on pre-computed distances and cosine similarities.

    Args:
        csv_filename (str): The path to the input CSV file.
    """
    if not os.path.exists(csv_filename):
        print(f"Error: File not found at '{csv_filename}'")
        return

    print(f"--- Loading data from '{csv_filename}' ---")
    # Load the data. Using float32 can save memory for large files.
    df = pd.read_csv(csv_filename, dtype={
        'query_id': np.int32,
        'neighbor_rank': np.int32,
        'neighbor_id': np.int64,
        'neighbor_cluster_id': np.int32,
        'c_q_dis': np.float32,
        'c_x_dis': np.float32,
        'cosine_similarity': np.float32
    })
    print(f"Successfully loaded {len(df)} records.")

    # --- 1. Pre-calculate required columns ---
    print("Calculating sqrt distances and ratio...")
    # Calculate sqrt in-place to save memory
    df['c_q_dist'] = np.sqrt(df['c_q_dis'])
    df['c_x_dist'] = np.sqrt(df['c_x_dis'])

    # Add a small epsilon to prevent division by zero
    df['dist_ratio'] = df['c_x_dist'] / (df['c_q_dist'] + 1e-9)

    # We no longer need the squared distance columns, drop them to free memory
    df.drop(columns=['c_q_dis', 'c_x_dis'], inplace=True)

    # --- 2. Group by neighbor_rank and aggregate ---
    print("Grouping by neighbor_rank and calculating statistics... (This may take a moment)")
    grouped = df.groupby('neighbor_rank')

    # --- MODIFICATION START ---

    # Define aggregation functions as a list of tuples:
    # Each tuple is ('output_column_name', aggregation_function)
    percentile_aggs = [
        ('cos_p01', lambda x: x.quantile(0.01)),
        ('cos_p05', lambda x: x.quantile(0.05)),
        ('cos_p10', lambda x: x.quantile(0.10)),
        ('cos_p25', lambda x: x.quantile(0.25)),
        ('cos_p50', lambda x: x.quantile(0.50)),  # Median
    ]

    # Calculate means for distance-based metrics as before
    analysis_df_means = grouped[['c_q_dist', 'c_x_dist', 'dist_ratio']].mean()

    # Apply the list of tuples for cosine similarity aggregation
    analysis_df_cos = grouped['cosine_similarity'].agg(percentile_aggs)

    # Combine into a single analysis DataFrame
    analysis_df = analysis_df_means.join(analysis_df_cos)

    print("Aggregation complete.")

    # --- 3. Visualize the results ---
    print("Generating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=True)
    fig.suptitle('Geometric Analysis of Neighbors within IVF Clusters (Averaged over all Queries)', fontsize=18)
    axes = axes.flatten()

    # Plot 1: Average Distance from Query to Neighbor's Centroid (sqrt(c_q_dis))
    axes[0].plot(analysis_df.index, analysis_df['c_q_dist'], color='blue')
    axes[0].set_title("Query Distance to Neighbor's Centroid", fontsize=14)
    axes[0].set_ylabel("Avg. L2 Distance", fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Average Distance from Neighbor to its Centroid (sqrt(c_x_dis))
    axes[1].plot(analysis_df.index, analysis_df['c_x_dist'], color='green')
    axes[1].set_title("Neighbor Distance to its own Centroid", fontsize=14)
    axes[1].set_ylabel("Avg. L2 Distance", fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Ratio of Distances (sqrt(c_x_dis) / sqrt(c_q_dis))
    axes[2].plot(analysis_df.index, analysis_df['dist_ratio'], color='red')
    axes[2].set_title("Ratio of Distances (Neighbor-to-Centroid / Query-to-Centroid)", fontsize=14)
    axes[2].set_xlabel("Neighbor Rank (0 = closest, 9999 = farthest)", fontsize=12)
    axes[2].set_ylabel("Avg. Ratio", fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.6)

    # Plot 4: Percentiles of Residual Cosine Similarity
    percentile_cols = analysis_df.columns[analysis_df.columns.str.startswith('cos_p')]
    for col in percentile_cols:
        percent = col.replace('cos_p', '')
        axes[3].plot(analysis_df.index, analysis_df[col], label=f"{percent}% percentile")

    axes[3].set_title("Percentiles of Residual Cosine Similarity", fontsize=14)
    axes[3].set_xlabel("Neighbor Rank (0 = closest, 9999 = farthest)", fontsize=12)
    axes[3].set_ylabel("Cosine Similarity", fontsize=12)
    axes[3].grid(True, linestyle='--', alpha=0.6)
    axes[3].legend()

    # Improve layout and save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_filename = "geometric_analysis_by_rank.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\nAnalysis complete. Visualization saved to '{output_filename}'")
    plt.close(fig)


if __name__ == "__main__":
    # --- IMPORTANT: Update this path to your actual file location ---
    CSV_FILE_PATH = "/mnt/E/共享/output.csv"

    analyze_and_visualize_geometry(CSV_FILE_PATH)