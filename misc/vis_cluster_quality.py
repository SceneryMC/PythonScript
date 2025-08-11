import numpy as np
import matplotlib

# Set non-interactive backend for server use. Must be before importing pyplot.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def visualize_clustering_quality(data_filename: str, output_filename: str = "clustering_quality_histogram.png"):
    """
    Reads a file of squared distances, analyzes their distribution,
    and saves a histogram of the actual L2 distances.

    Args:
        data_filename (str): Path to the input text file containing one squared distance per line.
        output_filename (str): Path to save the output PNG image.
    """
    if not os.path.exists(data_filename):
        print(f"Error: Data file not found at '{data_filename}'")
        return

    print(f"--- Analyzing Clustering Quality from '{data_filename}' ---")

    # 1. Load the data using numpy, which is very efficient for this task.
    print("Loading squared distances...")
    try:
        # dtype=np.float32 saves memory compared to the default float64
        squared_distances = np.loadtxt(data_filename, dtype=np.float32)
        print(f"Successfully loaded {len(squared_distances)} distance records.")
    except Exception as e:
        print(f"Error reading data file: {e}")
        return

    # 2. Calculate the actual L2 distances by taking the square root.
    # Handle potential negative values from floating point inaccuracies before sqrt.
    squared_distances[squared_distances < 0] = 0
    distances = np.sqrt(squared_distances)

    # 3. Print descriptive statistics to the console.
    print("\n--- Descriptive Statistics of L2 Distances ---")
    print(f"Mean Distance:              {np.mean(distances):.3f}")
    print(f"Standard Deviation:         {np.std(distances):.3f}")
    print(f"Min Distance:               {np.min(distances):.3f}")
    print(f"25th Percentile (Q1):       {np.percentile(distances, 25):.3f}")
    print(f"50th Percentile (Median):   {np.percentile(distances, 50):.3f}")
    print(f"75th Percentile (Q3):       {np.percentile(distances, 75):.3f}")
    print(f"95th Percentile:            {np.percentile(distances, 95):.3f}")
    print(f"99th Percentile:            {np.percentile(distances, 99):.3f}")
    print(f"Max Distance:               {np.max(distances):.3f}")
    print("-------------------------------------------------")

    # 4. Generate the histogram.
    print("\nGenerating histogram...")
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(distances, bins=100, color='c', edgecolor='black', alpha=0.75)

    ax.set_title('Distribution of Distances from Points to their Assigned Centroid', fontsize=16)
    ax.set_xlabel('L2 Distance to Centroid', fontsize=12)
    ax.set_ylabel('Number of Vectors (Frequency)', fontsize=12)

    # Using a logarithmic scale for the y-axis is often helpful
    # if many points are very close to the center.
    ax.set_yscale('log')
    ax.set_ylabel('Number of Vectors (Frequency - Log Scale)', fontsize=12)

    # Add vertical lines for mean and median to give context to the plot
    ax.axvline(np.mean(distances), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(distances):.2f}')
    ax.axvline(np.percentile(distances, 50), color='g', linestyle='-', linewidth=2,
               label=f'Median: {np.percentile(distances, 50):.2f}')

    ax.grid(True, which="both", linestyle='--', linewidth=0.5)
    ax.legend()

    # 5. Save the plot to a file.
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\nHistogram successfully saved to '{output_filename}'")
    plt.close(fig)


if __name__ == "__main__":
    # --- IMPORTANT: Update this path to your C++ output file ---
    # Assume your C++ program's output was redirected to this file.
    # Example C++ execution: ./my_program > clustering_distances.txt
    DATA_FILE_PATH = "/mnt/E/共享/output_cluster.csv"

    visualize_clustering_quality(DATA_FILE_PATH)