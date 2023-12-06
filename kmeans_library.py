from utils import create_data, visualize
import warnings
from sklearn.cluster import KMeans
from kmeans import KMeanClustering


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    data, label = create_data()
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
    print('Centers found by Scikit-Learn:')
    print(kmeans.cluster_centers_)

    print("==============================================================");
    model = KMeanClustering()
    labels, centroids = model.fit(data, iterations=300)
    print("Final Centroids without Scikit-Learn: ")
    print(centroids)

    model = KMeanClustering(k = 3);

    pred_label = kmeans.predict(data)
    visualize(data, label, kmeans.cluster_centers_)

