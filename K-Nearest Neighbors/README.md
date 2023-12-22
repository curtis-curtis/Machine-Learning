Based on Twitter US Airline Sentiment data, implemented a K-Nearest Neighbors regressor from scratch with NumPy to perform sentiment analysis.

- `train_dataset.npz` is a sparse matrix, which is further converted to a Numpy 2D array with shape $(900, 2642)$, where $900$ is the number of training data samples, and $2642$ is the dimension of each data point. The matrix represents the features of training data, which are related to the word frequency.

- `test_dataset.npz` is a sparse matrix, which is further converted to a Numpy 2D array with shape $(100, 2642)$, where $100$ is the number of test data samples, and $2642$ is the dimension of each data point. The matrix represents the features of test data, which are related to the word frequency.

- `train_labels.npy` is a Numpy 1D array with shape $(900, )$. The array represents the class labels for the training data samples, i.e., each element is an integer from $\{0,1,2\}$, representing the index of the sentiment class (positive, neutral, or negative).

- `test_labels.npy` is a Numpy 1D array with shape $(100, )$. The array represents the class labels for the text data samples, i.e., each element is an integer from $\{0,1,2\}$, representing the index of the sentiment class (positive, neutral, or negative).
