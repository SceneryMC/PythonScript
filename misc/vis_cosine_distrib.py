import pandas as pd
import numpy as np
import os


def analyze_cosine_distribution(csv_filename: str):
    """
    Analyzes the distribution of cosine similarities from the generated CSV file.

    Args:
        csv_filename (str): The path to the input CSV file.
    """
    if not os.path.exists(csv_filename):
        print(f"Error: File not found at '{csv_filename}'")
        return

    print(f"--- Loading and analyzing data from '{csv_filename}' ---")

    # Load the entire CSV into a pandas DataFrame.
    # For large files (e.g., 10000 queries * 1000 neighbors = 10M rows),
    # this might take a few moments and consume some memory.
    try:
        df = pd.read_csv(csv_filename)
        print(f"Successfully loaded {len(df)} records.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Ensure the column exists
    if 'cosine_similarity' not in df.columns:
        print("Error: 'cosine_similarity' column not found in the CSV file.")
        print(f"Available columns are: {df.columns.tolist()}")
        return

    # --- 1. Overall Distribution Analysis ---
    print("\n--- Overall Distribution of Cosine Similarities (All Neighbors) ---")

    # Use pandas' describe() for a quick and comprehensive summary
    overall_stats = df['cosine_similarity'].describe(percentiles=[.01, .1, .25, .5, .75, .9, .99])
    print(overall_stats.to_string())

    # Calculate the mean of absolute values
    mean_abs_cosine = df['cosine_similarity'].abs().mean()
    print(f"\nMean of Absolute Cosine Values: {mean_abs_cosine:.6f}")

    # Calculate the percentage of "nearly orthogonal" vectors
    orthogonality_threshold = 0.1
    nearly_orthogonal_pct = (df['cosine_similarity'].abs() < orthogonality_threshold).mean() * 100
    print(f"Percentage of vectors with |cos(theta)| < {orthogonality_threshold}: {nearly_orthogonal_pct:.2f}%")

    # --- 2. Distribution Analysis by Neighbor Rank ---
    print("\n--- Distribution Stratified by Neighbor Rank ---")

    # Analyze the very first nearest neighbor (rank 0)
    print("\n--- Stats for the 1st Nearest Neighbor (rank=0) ---")
    rank_0_stats = df[df['neighbor_rank'] == 0]['cosine_similarity'].describe()
    print(rank_0_stats.to_string())

    # Analyze neighbors ranked 1 to 9
    print("\n--- Stats for Neighbors ranked 1-9 ---")
    rank_1_9_stats = df[(df['neighbor_rank'] >= 1) & (df['neighbor_rank'] <= 9)]['cosine_similarity'].describe()
    print(rank_1_9_stats.to_string())

    # Analyze neighbors ranked 10 to 99
    print("\n--- Stats for Neighbors ranked 10-99 ---")
    rank_10_99_stats = df[(df['neighbor_rank'] >= 10) & (df['neighbor_rank'] <= 99)]['cosine_similarity'].describe()
    print(rank_10_99_stats.to_string())

    # Analyze neighbors from the middle of the pack (e.g., 450-549) if k is large enough
    if df['neighbor_rank'].max() >= 549:
        print("\n--- Stats for Neighbors ranked 450-549 ---")
        rank_mid_stats = df[(df['neighbor_rank'] >= 450) & (df['neighbor_rank'] <= 549)]['cosine_similarity'].describe()
        print(rank_mid_stats.to_string())

    # Analyze the farthest neighbors in the ground truth set
    farthest_rank_start = int(df['neighbor_rank'].max() * 0.9)  # Last 10%
    print(f"\n--- Stats for Farthest Neighbors (rank >= {farthest_rank_start}) ---")
    rank_far_stats = df[df['neighbor_rank'] >= farthest_rank_start]['cosine_similarity'].describe()
    print(rank_far_stats.to_string())

    print("\n--- Analysis Summary ---")
    print("Interpretation:")
    print(
        "1. If the OVERALL mean is very close to 0 and the std is small, the 'concentration around 90 degrees' hypothesis is supported.")
    print(
        "2. Compare the MEAN values across different rank groups. If the mean for rank=0 is significantly higher than for farther ranks, it indicates that the true nearest neighbor has a special geometric relationship that is lost for more distant points.")


if __name__ == "__main__":
    # --- IMPORTANT: Update this path to your actual file location ---
    CSV_FILE_PATH = "/mnt/E/共享/output.csv"

    analyze_cosine_distribution(CSV_FILE_PATH)