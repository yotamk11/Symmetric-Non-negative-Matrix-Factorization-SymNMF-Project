import sys
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import symnmf
import kmeans
max_iter = 300
epsilon = 1e-4

def get_labels(data, centroids):
    """Assigns each point to the closest centroid to get cluster labels."""
    labels = []
    for point in data:
        min_dist = float('inf')
        label = 0
        for i, centroid in enumerate(centroids):
            dist = kmeans.euclidean_distance(point, centroid)
            if dist < min_dist:
                min_dist = dist
                label = i
        labels.append(label)
    return labels

def main():
    if len(sys.argv) < 3:
        print("An Error Has Occurred")
        sys.exit(1)
    
    k = int(sys.argv[1])
    file_name = sys.argv[2]

    

    # 1. Load Data
    data_points = pd.read_csv(file_name, header=None).values.tolist()
    n = len(data_points)
    if (k > n or k <= 0):
        raise ValueError("Invalid number of clusters")
    d = len(data_points[0])


    # 2. SymNMF Clustering
    # Get normalized matrix W
    W = symnmf.norm(data_points, n, d)
    
    # Initialize H (Section 1.4.1)
    np.random.seed(1234)
    m = np.mean(W)
    upper_bound = 2 * np.sqrt(m / k)
    initial_H = np.random.uniform(0, upper_bound, (n, k)).tolist()
    
    # Run SymNMF in C
    final_H = np.array(symnmf.symnmf(W, initial_H, n, k))
    # Assign cluster by max value in row (Section 1.5)
    nmf_labels = np.argmax(final_H, axis=1)

    # 3. KMeans Clustering (Using your homework file)
    # Note: Kmeans from HW uses first K points as initial centroids
    final_centroids = kmeans.kmeans_alg(k, data_points, max_iter, epsilon)
    kmeans_labels = get_labels(data_points, final_centroids)

    # 4. Calculate Scores
    nmf_score = silhouette_score(data_points, nmf_labels)
    km_score = silhouette_score(data_points, kmeans_labels)
        
    # 5. Output Results (Section 1.5)
    print(f"nmf: {nmf_score:.4f}")
    print(f"kmeans: {km_score:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)