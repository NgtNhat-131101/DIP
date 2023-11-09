import numpy as np
import matplotlib.pyplot as plt
from utils import create_data, visualize
from scipy.spatial.distance import cdist

class KMeanClustering:

    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    def init_centroids(self, X):
        return X[np.random.choice(X.shape[0], self.k, replace=False)]
    
    def euclidean_dis(self, X):
        D = cdist(X, self.centroids)
        return np.argmin(D, axis=1)
    
    def update_centroids(self, data, label):
        new_centroids = np.zeros((self.k, data.shape[1]))
        
        for cluster in range(self.k):
            xk = data[label == cluster, :]
            if xk.size > 0:
                new_centroids[cluster, :] = np.mean(xk, axis=0)

        return new_centroids
    
    def has_converged(self, new_centers):
        return np.array_equal(self.centroids, new_centers)

    def fit(self, data, iterations=100):
        self.centroids = self.init_centroids(data)
        labels = []

        for i in range(iterations):
            labels.append(self.euclidean_dis(data))
            new_centroids = self.update_centroids(data, labels[-1])
            if self.has_converged(new_centroids):
                break
            self.centroids = new_centroids

        return labels, self.centroids

    def predict(self, data):
        return self.euclidean_dis(data)


if __name__ == "__main__":
    data, label = create_data()
    k = 3

    Kmeans = KMeanClustering(k=k)
    centroids = Kmeans.init_centroids(data)
    # print("Centroids without scikit-learn: ", centroids)
    visualize(data, label, centroids)
    labels, centroids = Kmeans.fit(data, iterations=300)
    print("Centroids without scikit-learn: ")
    print(centroids)
    # visualize(data, label, centroids)
    visualize(data, labels[-1], centroids)