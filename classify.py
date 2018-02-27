from sklearn import decomposition
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def multinomial_nb(x_train, y_train, x_test, y_test):
  mnb = MultinomialNB()
  mnb.fit(x_train, y_train)
  return mnb.score(x_test, y_test)

def gaussian_nb(x_train, y_train, x_test, y_test):
  gnb = GaussianNB()
  gnb.fit(x_train, y_train)
  return gnb.score(x_test, y_test)

def linear_svc(x,y):
  lsvc = LinearSVC(random_state=0)
  return cross_val_score(lsvc, x, y, cv = 10, scoring = 'accuracy')

def k_nearest_neighbors(x_train, y_train, x_test, y_test):
  knn = KNeighborsClassifier(n_neighbors=3)
  knn.fit(x_train, y_train)
  return knn.score(x_test, y_test)

def logistic_regression(x_train, y_train, x_test, y_test):
  lr = LogisticRegression()
  lr.fit(x_train, y_train)
  return lr.score(x_test, y_test)



if __name__ == '__main__':
  x_train = pd.read_csv(filepath_or_buffer='train_bag_of_words_5.csv', header=None, sep=',').values
  y_train = pd.read_csv(filepath_or_buffer='train_classes_5.txt', header=None, sep=',').values
  x_test = pd.read_csv(filepath_or_buffer='test_bag_of_words_0.csv', header=None, sep=',').values
  y_test = pd.read_csv(filepath_or_buffer='test_classes_0.txt', header=None, sep=',').values


  x = np.concatenate((x_train,x_test), axis=0)
  y = np.concatenate((y_train,y_test), axis=0).ravel()
  print("read in data")

  print(linear_svc(x,y))
  # cross_validation(x_train, y_train, x_test, y_test)
  # print(logistic_regression(x_train,y_train,x_test,y_test))
  # print(k_nearest_neighbors(x_train, y_train, x_test, y_test))
  # print(gaussian_nb(x_train, y_train, x_test, y_test))
