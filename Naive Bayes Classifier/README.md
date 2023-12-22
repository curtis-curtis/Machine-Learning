This is a sentiment analysis on Twitter US Airline Sentiment data with Naive Bayes Classifier, implemented by NumPy.

* `train_dataset.npy` is a Numpy 2D array with shape  (900,2642) , where  900  is the number of training data samples, and  2642  is the number of features. The elements in the array are in Boolean data type.

* `test_dataset.npy` is a Numpy 2D array with shape  (100,2642) , where  100  is the number of test data samples, and  2642  is the number of features. The elements in the array are in Boolean data type.

* `train_labels.npy` is a Numpy 1D array with shape  (900,) . The array represents the class labels for the training data samples, i.e., each element is an integer from  {0,1,2} , representing the index of the sentiment class (positive, neutral, or negative).

* `test_labels.npy` is a Numpy 1D array with shape  (100,) . The array represents the class labels for the text data samples, i.e., each element is an integer from  {0,1,2} , representing the index of the sentiment class (positive, neutral, or negative).
