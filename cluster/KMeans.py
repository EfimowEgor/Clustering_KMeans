from typing import Any, Optional

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from services.gif_service import IGifCreator
from providers.plt_3d_gif import Matplotlib3DGifCreator
from providers.plt_2d_gif import Matplotlib2DGifCreator

from graphviz import Digraph


# np.random.seed(12)


class KMeans:
    def __init__(
        self,
        n_clusters: int,
        *,
        max_iters: int = 300,
        tol: float = 1e-4,
        gif_creator: Optional[IGifCreator] = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.X = None
        self.centroids = None
        self.labels = None
        self.gif_creator = gif_creator

    def __init_centroids(self) -> None:
        n_features = self.X.shape[1]
        self.centroids = np.zeros((self.n_clusters, n_features))

        for i in range(self.n_clusters):
            for j in range(n_features):
                min_val = np.min(self.X[:, j])
                max_val = np.max(self.X[:, j])
                self.centroids[i, j] = np.random.uniform(min_val, max_val)

    def fit(self, X: np.ndarray) -> "KMeans":
        self.X = X
        self.__init_centroids()
        return self

    def __update(self):
        new_centroids = np.zeros_like(self.centroids)

        for i in range(self.n_clusters):
            cluster_points = self.X[self.labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                new_centroids[i] = self.centroids[i]

        assert new_centroids.shape == self.centroids.shape
        return new_centroids

    def __assign_clusters(self) -> np.ndarray:
        distances = np.linalg.norm(
            self.X[:, np.newaxis, :] - self.centroids, axis=2
        )
        return np.argmin(distances, axis=1)

    def predict(self) -> np.ndarray:
        total_errors = []

        for iter in range(self.max_iters):
            old_centroids = self.centroids
            self.labels = self.__assign_clusters()
            self.centroids = self.__update()

            total_error = 0
            for i, centroid in enumerate(self.centroids):
                cluster_points = self.X[self.labels == i]
                distances = np.linalg.norm(cluster_points[:, np.newaxis, :] - centroid, axis=2)
                total_error += np.sum(distances ** 2)
            total_errors.append(total_error)

            if self.gif_creator:
                self.gif_creator.add_frame(
                    self.X, self.labels, self.centroids, iter + 1
                )

            if np.allclose(old_centroids, self.centroids, rtol=self.tol):
                break

        if self.gif_creator:
            self.gif_creator.save()

        return self.labels, total_errors
    
    def class_tree(self, labels_by_k: dict[int, np.ndarray[Any, Any]], object_names=None):
        dot = Digraph(comment='class_tree')
        max_k = max(labels_by_k.keys())

        for i, obj_name in enumerate(object_names):
            dot.node(f"obj_{i}", label=obj_name)

        dot.node("root", label="All data (k=1)")

        for i, label in enumerate(labels_by_k[max_k]):
            dot.node(f"cluster_{max_k}_{label}", label=f"Cluster {label} (k={max_k})")
            dot.edge(f"obj_{i}", f"cluster_{max_k}_{label}")

        for k in range(2, max_k + 1):
            for cluster_id in range(k):
                dot.node(f"cluster_{k}_{cluster_id}", label=f"Cluster {cluster_id} (k={k})")

            if k == 2:
                for cluster_id in range(k):
                    dot.edge(f"cluster_{k}_{cluster_id}", "root")

            if k > 2:
                for parent_id in range(k - 1):
                    for child_id in range(k):
                        if any(
                            labels_by_k[k][labels_by_k[k - 1] == parent_id]
                            == child_id
                        ):
                            dot.edge(
                                f"cluster_{k}_{child_id}",
                                f"cluster_{k-1}_{parent_id}",
                            )
        dot.render("./P5/kmeans_tree", format="png", cleanup=True)



if __name__ == "__main__":
    # wine-clustering.csv
    # read_excel("./P5/data.xlsx", header=None)
    df = pd.read_excel("./P5/data.xlsx", header=None)

    df = df.values

    # pca = PCA(n_components=2)
    # pca_data = pca.fit_transform(df)

    names = ["feat1", "feat2", "feat3"]

    labels_by_k = {}
    err = []

    for n_cluster in range(2, 5):
        gif_creator = Matplotlib2DGifCreator(filename=f"./P5/{n_cluster}_update_kmeans.gif", fps=1, names=names)
        kmeans = KMeans(n_clusters=n_cluster, gif_creator=gif_creator)
        kmeans.fit(df)
        labels, errors = kmeans.predict()
        errors = [val.item() for val in errors]
        err.append(errors[-1])

        print(err, len(err))

        labels_by_k[n_cluster] = np.array(labels)

    kmeans.class_tree(labels_by_k, [])

    plt.scatter(x=np.arange(0, len(err)), y=err)
    plt.plot(err)
    plt.show()