from utils import create_data, visualize
import warnings
from sklearn.cluster import KMeans


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    data, label = create_data()
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
    print('Centers found by scikit-learn:')
    print(kmeans.cluster_centers_)
    pred_label = kmeans.predict(data)
    visualize(data, label, kmeans.cluster_centers_)