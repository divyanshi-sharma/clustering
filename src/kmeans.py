import numpy as np
from scipy.spatial import distance

class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        distances = np.ones((features.shape[0], self.n_clusters))
        old_labels = np.ones(features.shape[0])
        new_labels = np.zeros(features.shape[0])


        centroids = np.array([])
        sample = np.random.permutation(range(features.shape[0]))
        for i in range(self.n_clusters):
            centroids = np.append(centroids, (features[sample[i], :]))
        new_means = centroids.reshape(self.n_clusters, features.shape[1])
        max_iters = 10

        def update_labels(mean, feat):
            for j in range(self.n_clusters):
                for l in range(feat.shape[0]):
                    distances[l, j] = distance.euclidean(feat[l, :], mean[j, :])
            new_label = np.argmin(distances, axis=1)
            return new_label

        def update_means(new_label, feat):
            sums = np.zeros((self.n_clusters, features.shape[1]))
            for k in range(self.n_clusters):
                indices = np.where(new_label == k)
                indices = indices[0]
                if indices != []:
                    for j in range(len(indices)):
                        sums[k, :] += feat[indices[j], :]
                    new_means[k, :] = sums[k, :]/len(indices)
            return new_means

        while not np.allclose(old_labels, new_labels) and max_iters > 0:
            old_labels = new_labels
            new_labels = update_labels(new_means, features)
            new_means = update_means(new_labels, features)
            max_iters -= 1

        self.means = new_means

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        distances = np.ones((features.shape[0], self.n_clusters))
        for j in range(self.n_clusters):
            for l in range(features.shape[0]):
                distances[l, j] = distance.euclidean(features[l, :], self.means[j, :])
        predictions = np.argmin(distances, axis=1)
        return predictions
