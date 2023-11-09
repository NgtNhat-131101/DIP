import numpy as np
import matplotlib.pyplot as plt
np.random.seed(100)
def visualize(data, label, centroids):
    k = np.amax(label) + 1

    for cluster in range(k):
        X = data[label == cluster]
        centroid = centroids[cluster]
        marker = 's' if cluster == 0 else ('^' if cluster == 1 else 'o')
        color = 'hotpink' if cluster == 0 else ('yellow' if cluster == 1 else 'black')
        label_text = f'Centroid {cluster}'

        plt.plot(X[:, 0], X[:, 1], marker, markersize=4, alpha=0.8, label=f'Class {cluster}')
        plt.scatter(centroid[0], centroid[1], s=300, color=color, marker=marker, label=label_text, linewidths=2, edgecolors='black')

    plt.title("Data Visualization")
    plt.legend()
    plt.axis('off')
    plt.show()

def visualize_3d(data, label, centroids):
    k = np.amax(label) + 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D plot

    for cluster in range(k):
        X = data[label == cluster]
        centroid = centroids[cluster]
        marker = 's' if cluster == 0 else ('^' if cluster == 1 else 'o')
        color = 'hotpink' if cluster == 0 else ('yellow' if cluster == 1 else 'black')
        label_text = f'Centroid {cluster}'

        ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker=marker, s=40, color=color, label=f'Class {cluster}')
        ax.scatter(centroid[0], centroid[1], centroid[2], s=200, color=color, marker=marker, label=label_text, edgecolors='black')

    ax.set_title("Data Visualization")
    ax.legend()
    plt.show()

def create_data():
    means = [[2, 2], [8, 3], [3, 6]]
    cov = [[1, 0], [0, 1]]
    N = 500   
    X0 = np.random.multivariate_normal(means[0], cov, N)
    X1 = np.random.multivariate_normal(means[1], cov, N)
    X2 = np.random.multivariate_normal(means[2], cov, N)
    
    label = np.asarray([0]*N + [1]*N + [2]*N)
    X = np.concatenate((X0, X1, X2), axis=0)
    return X, label