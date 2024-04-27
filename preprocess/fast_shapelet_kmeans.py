import numpy as np
from sklearn.cluster import KMeans

def distance(ts1, ts2):
    # Euclidean distance
    return np.sqrt(np.sum((ts1 - ts2) ** 2))

def fast_shapelets(data, window_length, num_shapelet, max_dist=float('inf')):
    """
    This function implements a simplified version of Fast Shapelets

    Args:
        data: A numpy array of shape (n_samples, time_series_length)
        window_length: Maximum length of shapelets to consider
        max_dist: Threshold for average distance (higher = less strict)

    Returns:
        A list of significant shapelets and their corresponding average distances
    """
    best_shapelets = []
    best_dists = []
    for i in range(len(data)):
        for window_start in range(len(data[i]) - window_length + 1):
            # Extract potential shapelet
            shapelet = data[i, window_start:window_start+window_length]
            # Calculate average distance to other classes
            dists = [distance(shapelet, ts[window_start:window_start+window_length])
                     for ts in data if ts is not data[i]]
            if dists:  # Check if dists is not empty
                avg_dist = np.mean(dists)
                if avg_dist < max_dist and (not best_dists or avg_dist < best_dists[0]):
                    best_shapelets = [shapelet]
                    best_dists = [avg_dist]
                elif avg_dist < max_dist and len(best_shapelets) < num_shapelet:  # Allow keeping top 3
                    best_shapelets.append(shapelet)
                    best_dists.append(avg_dist)
    # Return top shapelets
    return best_shapelets, best_dists


def fast_shapelet_kmeans(h_td, shapelet_length, num_shapelets, num_clusters):
# Assuming h_td is your tensor containing differences for each station
    h_td_np = h_td.numpy()

    # Initialize lists to store shapelets for each row
    shapelets_per_row = []

    # Extract shapelets for each row using fast_shapelets function
    for row in h_td_np:
        shapelets, _ = fast_shapelets(np.expand_dims(row, axis=0), window_length=shapelet_length, num_shapelet=num_shapelets)
        shapelets_per_row.append(shapelets)

    # Reshape the shapelets_per_row to (num_stations, num_shapelets * shapelet_length)
    feature_matrix = np.array(shapelets_per_row).reshape(len(h_td_np), -1)

    # Apply k-means clustering to classify the stations
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(feature_matrix)
    return kmeans.labels_