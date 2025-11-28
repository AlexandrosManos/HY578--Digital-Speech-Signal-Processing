import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def train_synthetic_gmm():
    """
    Generates synthetic 2D data, trains a GMM, and plots the clusters.
    """
    # 1. Generate synthetic data
    np.random.seed(42)
    
    # Cluster 1
    mean1 = [2, 2]
    cov1 = [[1, 0.5], [0.5, 1]]
    data1 = np.random.multivariate_normal(mean1, cov1, 200)
    
    # Cluster 2
    mean2 = [7, 7]
    cov2 = [[1, -0.6], [-0.6, 1]]
    data2 = np.random.multivariate_normal(mean2, cov2, 200)
    
    # Cluster 3
    mean3 = [8, 2]
    cov3 = [[0.5, 0], [0, 0.5]]
    data3 = np.random.multivariate_normal(mean3, cov3, 200)
    
    X = np.vstack([data1, data2, data3])
    
    # 2. Train GMM using EM (via scikit-learn)
    # We use 3 components because we generated 3 clusters
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    gmm.fit(X)
    
    # Predict clusters
    labels = gmm.predict(X)
    
    # 3. Plotting
    plt.figure(figsize=(10, 8))
    
    # Plot data points colored by cluster
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap='viridis', alpha=0.6)
    
    # Plot the centers
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', s=100, marker='x', label='Centroids')
    
    plt.title('GMM Clustering on Synthetic Data (EM Algorithm)')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = 'synthetic_gmm_plot.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
    # Print parameters
    print("GMM Means:\n", gmm.means_)
    print("GMM Covariances shape:", gmm.covariances_.shape)

if __name__ == "__main__":
    train_synthetic_gmm()
