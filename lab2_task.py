# Hung Tsz Shing
class NaiveBayesClassifier:
  def __init__(self):
    self.train_dataset = None
    self.train_labels = None
    self.train_size = 0
    self.num_features = 0
    self.num_classes = 0

  def fit(self, train_dataset, train_labels):
    self.train_dataset = train_dataset
    self.train_labels = train_labels
    # TODO
    self.train_size = train_dataset.shape[0]
    self.num_features = train_dataset.shape[1]
    self.num_classes = np.amax(train_labels) + 1
  
  def estimate_class_prior(self):
    # TODO
    all_class = np.arange(self.num_classes) # [0, 1, 2]
    class_position = all_class == np.array([self.train_labels]).T # shape(900,3)
    class_count = np.sum(class_position, axis=0)
    class_prior = (class_count + 1)/(self.train_size + self.num_classes)
    return class_prior
    # return class_prior

  def estimate_likelihoods(self):
    # TODO
    all_class = np.arange(self.num_classes) # [0, 1, 2]
    class_position = all_class == np.array([self.train_labels]).T # shape(900,3)
    frequency_features_yes = self.train_dataset.T @ class_position.astype(int) # shape(2642,3)
    class_count = np.sum(class_position, axis=0) # shape(3,)
    likelihoods = (frequency_features_yes + 1)/(class_count + 2)
    return likelihoods
    # return likelihoods

  def predict(self, test_dataset): # test_dataset shape(100,2642)
    class_prior = self.estimate_class_prior()
    yes_likelihoods = self.estimate_likelihoods()
    no_likelihoods = 1 - yes_likelihoods
    # TODO
    sum_true = test_dataset @ np.log(yes_likelihoods) # (100,3)
    sum_false = (1 - test_dataset) @ np.log(no_likelihoods)
    total_sum = sum_true + sum_false + np.log(class_prior)
    test_predict = np.argmax(total_sum, axis=1)
    return test_predict