import numpy as np
from scipy import stats
from sklearn import cluster, datasets
from sklearn.preprocessing import KBinsDiscretizer


class KPrototypes:
    def __init__(self, k, X, n_features, max_iter=100):
        """
        Initialization function for KPrototypes class. It is called every time an object is created from this class.
        :param k: int, required
        The number of clusters to form as well as the number of prototypes to generate.
        :param X: ndarray of shape (n_samples, n_features), required
        Training instances to cluster.
        :param n_features: int, required
        The number of dimensions of each data instance as well as the number of features. Including both categorical and
        numerical features.
        :param max_iter: int, default=100
        The maximum number of iterations of the K-prototypes algorithm for a single run.
        """
        self.k = k
        self.n_features = n_features
        self.max_iter = max_iter

        # TODO: Randomly select k data points from X as the initial prototypes.
        # Hint: Do not hardcode the indices of selected data points.
        # You may use Numpy's 'random.randint' function to randomly choose the initial prototypes.
        selected_idx = np.random.randint(X.shape[0], size=k)  # X.shape[0] is n_samples          # ndarray of shape (k, )
        self.prototypes = X[selected_idx]                                                        # ndarray of shape (k, n_features)

    def euclidean_distance(self, X, is_categ, debug_prototypes=None):
        """
        Calculate the Euclidean distance between each data point and prototypes. Only consider the numerical feature
        columns of each data point and prototypes.
        :param X: ndarray of shape (n_samples, n_features), required
        Training instances to cluster.
        :param is_categ: ndarray of shape (n_features,), required
        Array of boolean values indicates whether each feature column is categorical or not.
        e.g., is_categ=[False, False, True, True] indicates that feature 2 and feature 3 of dataset X are categorical.
        Then, the remaining feature columns (i.e., feature 0 and feature 1) are numerical.
        :param debug_prototypes: ndarray of shape (k, n_features), optional
        If debug_prototypes is given, it will be used in the calculation rather than the stored prototypes in the Class
        Object. This argument is only used to help you test the function independently. Do not use this argument in the
        fit_predict function.

        :return: ndarray of shape (n_prototypes, n_samples)
        The Euclidean distance between each data point and prototypes.
        """
        prototypes = debug_prototypes if debug_prototypes is not None else self.prototypes
        # TODO: only use the numerical feature columns to calculate the Euclidean distance between each data point and
        #       prototypes.
        # Hints:
        #   - Notice that X and prototypes have mismatched shapes.
        #     X: (n_samples, n_features)                    prototypes: (n_prototypes, n_features)
        #   - Only the numerical feature columns are used to calculate the Euclidean distance.
        #     X: (n_samples, n_numerical_features)          prototypes: (n_prototypes, n_numerical_features)
        #   - You may use Numpy's 'count_nonzero', 'reshape', and 'linalg.norm' (or 'sqrt' & 'sum') functions.
        #   - Try broadcasting.
        n_numer_features = np.sum(1 - is_categ)  # OR = X.shape[1] - np.sum(is_categ)    1 - is_categ is ~is_categ           # Optional: number of numerical feature columns

        numer_prototypes = prototypes[:, ~is_categ]  # shape is (n_prototypes, n_numer_features)
        numer_X = X[:, ~is_categ]

        reshaped_numer_prototypes = np.reshape(numer_prototypes, (self.k, 1, n_numer_features))                                                                # Optional: reshape the numerical prototypes so that we can do broadcasting later
        difference = numer_X - reshaped_numer_prototypes  # shape is (self.k, n_numer_features, X.shape[0])                                                                             # Optional: use broadcasting to calculate the difference
        dist = np.linalg.norm(difference, axis=2)                                                                          # Required

        return dist

    def hamming_distance(self, X, is_categ, debug_prototypes=None):
        """
        Calculate the Hamming distance between each data point and prototypes. Only consider the categorical feature
        columns of each data point and prototypes.
        :param X: ndarray of shape (n_samples, n_features), required
        Training instances to cluster.
        :param is_categ: ndarray of shape (n_features,), required
        Array of boolean values indicates whether each feature column is categorical or not.
        e.g., is_categ=[False, False, True, True] indicates that feature 2 and feature 3 of dataset X are categorical.
        Then, the remaining feature columns (i.e., feature 0 and feature 1) are numerical.
        :param debug_prototypes: ndarray of shape (k, n_features), optional
        If debug_prototypes is given, it will be used in the calculation rather than the stored prototypes in the Class
        Object. This argument is only used to help you test the function independently. Do not use this argument in the
        fit_predict function.

        :return: ndarray of shape (n_prototypes, n_samples)
        The Hamming distance between each data point and prototypes.
        """
        prototypes = debug_prototypes if debug_prototypes is not None else self.prototypes
        # TODO: only use the categorical feature columns to calculate the Hamming distance between each data point and
        #       prototypes.
        # Hints:
        #   - Notice that X and prototypes have mismatched shapes.
        #     X: (n_samples, n_features)                    prototypes: (n_prototypes, n_features)
        #   - Only the categorical feature columns are used to calculate the Hamming distance.
        #     X: (n_samples, n_categorical_features)        prototypes: (n_prototypes, n_categorical_features)
        #   - You may use Numpy's 'count_nonzero', 'reshape', 'sum', and 'not_equal' functions.
        #   - Try broadcasting.
        n_categ_features = np.sum(is_categ)                                         # Optional: number of categorical feature columns
        categ_X = X[:, is_categ]
        categ_prototypes = prototypes[:, is_categ]
        reshaped_categ_prototypes = np.reshape(categ_prototypes, (self.k, 1, n_categ_features))            # Optional: reshape the categorical prototypes so that we can do broadcasting later
        difference = categ_X != reshaped_categ_prototypes                                    # Optional: use broadcasting to calculate the difference
        dist = np.sum(difference, axis=2)                                                        # Required

        return dist

    def fit_predict(self, X, is_categ):
        '''
        Compute cluster centers and predict cluster index for each sample.
        :param X: ndarray of shape (n_samples, n_features), required
        Training instances to cluster.
        :param is_categ: ndarray of shape (n_features,), required
        Array of boolean values indicates whether each feature column is categorical or not.
        e.g., is_categ=[False, False, True, True] indicates that feature 2 and feature 3 of dataset X are categorical.
        Then, the remaining feature columns (i.e., feature 0 and feature 1) are numerical.

        :return: ndarray of shape (n_samples,)
        Index of the cluster (serve as the label) each sample belongs to.
        '''
        prev_prototypes = None
        iteration = 0

        # TODO: Set the criterion to leave the loop.
        # Hints:
        #   - The criterion to leave the loop is to satisfy either of the two conditions:
        #     1. Convergence criterion: the prototypes are the same as those in the last iteration.
        #     2. Max number of iterations: the algorithm runs to the max number of iterations, i.e., self.max_iter
        #   - You may use Numpy's 'not_equal' and 'any' function.
        while np.any(prev_prototypes != self.prototypes) and iteration < self.max_iter:

            # TODO: Assign the index of the closest prototype to each data point.
            # Hints: You may use numpy.argmin function to find the index of the closest prototype for each data point.
            numer_dist = self.euclidean_distance(X, is_categ)
            categ_dist = self.hamming_distance(X, is_categ)
            dist = numer_dist + categ_dist
            prototype_idx = np.argmin(dist, axis=0)

            prev_prototypes = self.prototypes.copy()  # Push current prototypes to previous.

            # TODO: Reassign prototypes as the mean of the clusters.
            # Hints:
            #  - We mentioned a method to choose specific elements from an array.
            #  - We mentioned that there were lots of functions from NumPy or scipy.stats for statistics.
            #    mean, std, median, mode, etc. On what axis should we find the statistics?
            #  - 'np.mean' and 'stats.mode' has different return shape. See how 'np.squeeze' works.
            for i in range(self.k):
                # A boolean array of shape (n_samples,). e.g., [False, True] means the second data sample is assigned to
                # cluster i but the first data sample is not.
                assigned_idx = prototype_idx == i                 #np.full(X.shape[0], i)
                
                if np.count_nonzero(assigned_idx) == 0:
                    continue

                if np.count_nonzero(~is_categ) > 0: # numerical value more than 0
                    # Update the prototypes
                    self.prototypes[i, ~is_categ] = np.mean(X[assigned_idx][:, ~is_categ], axis=0)

                if np.count_nonzero(is_categ) > 0: # categorical value more than 0
                    # The returned ndarray of stats.mode does not have the same shape as the 'np.mean' function.
                    # Please set the keepdims parameter of stats.mode function to avoid Warning. 
                    categ_mode, _ = stats.mode(X[assigned_idx][:, is_categ], keepdims=False)

                    # Convert this returned ndarray to the same shape as the 'np.mean' function before updating the prototypes.
                    self.prototypes[i, is_categ] = categ_mode

            iteration += 1
        return prototype_idx


def SSE(X, y, k, centroids):
    sse = 0
    # TODO: For each cluster, calculate the distance (square of difference, i.e. Euclidean/L2-distance) of samples to
    #  the datapoints and accumulate the sum to `sse`. (Hints: use numpy.sum and for loop)
    # Hints:
    #   - X is a Numpy 2D array with shape (num_datapoints, ndim), representing the data points.
    #   - y is a Numpy 1D array with shape (num_datapoints, ), representing which cluster (or which centroid) each data
    #   point correspond to.
    #   - This is very similar to the distance functions in Task 1
    for i in range(k):
        assigned_idx = y == np.full(X.shape[0], i)
        difference = X[assigned_idx] - centroids[i]
        sse += np.sum(difference * difference)
    return sse
