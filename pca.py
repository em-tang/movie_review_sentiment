from sklearn import decomposition
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# def naive_bayes:


def k_nearest_neighbors(x_train, y_train, x_test, y_test):
# knn = KNeighborsClassifier(n_neighbors=3)
  return

def logistic_regression(x_train, y_train, x_test, y_test):
  lr = LogisticRegression()
  lr.fit(x_train, y_train)
  return lr.score(x_test, y_test)

def svm():
  y = np.zeros(2400)
  y[1201:] = 1
  mean_vec = np.mean(x, axis=0)
  cov_mat = (x - mean_vec).T.dot((x - mean_vec)) / (x.shape[0]-1)
  print('Covariance matrix \n%s' %cov_mat)
  # correlation = np.corrcoef(cov_mat.T)
  evals, evecs = np.linalg.eig(cov_mat)


  # Make a list of (eigenvalue, eigenvector) tuples
  epairs = [(np.abs(evals[i]), evecs[:,i]) for i in range(len(evals))]

  # Sort the (eigenvalue, eigenvector) tuples from high to low
  epairs.sort()
  epairs.reverse()

  # Visually confirm that the list is correctly sorted by decreasing eigenvalues
  print('Eigenvalues in descending order:')
  for i in epairs:
        print(i[0])

# def 
if __name__ == '__main__':
  x_train = pd.read_csv(filepath_or_buffer='out_bag_of_words_5.csv', header=None, sep=',').values

  y_train = pd.read_csv(filepath_or_buffer='out_classes_5.txt', header=None, sep=',').values
  x_test = pd.read_csv(filepath_or_buffer='test_bag_of_words_5.csv', header=None, sep=',').values
  y_test = pd.read_csv(filepath_or_buffer='test_classes_5.txt', header=None, sep=',').values
  
  train_txt = open("out_vocab_5.txt", "r")
  train_words = train_txt.read().split('\n')
  test_txt = open("test_vocab_5.txt", "r")
  test_words = test_txt.read().split('\n')
  intersection = list(set(test_words) & set(train_words))
  x_test = 
  
  # test_words = [] 
  # for i in range(train_words):
    # if word not in test_words_unclean: 
      # test_words.append(0)
    # else:
      # test_words.append(x_te


  print("read in data")
  # print(logistic_regression(x_train,y_train,x_test,y_test))
