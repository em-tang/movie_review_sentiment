from sklearn import decomposition
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def decision_tree(x_train, y_train, x_test, y_test):
  clf = DecisionTreeClassifier(random_state=0)
  clf.fit(x_train, y_train)
  pred = clf.predict(x_test)
  print(precision_recall_fscore_support(y_test, pred, average = 'binary'))
  return pred

def ensemble_decision_tree(x_train, y_train, x_test, y_test):
  x_t_a, x_t_b, a, b  = np.split(x_train,4)
  y_t_a, y_t_b, y_a, y_b = np.split(y_train,4)
  # mix it up a bit so that it doesn't only train on positive/negative reviews
  x_train_a = np.concatenate((x_t_a, a), axis=0)
  x_train_b = np.concatenate((x_t_b, b), axis=0)
  y_train_a = np.concatenate((y_t_a,y_a), axis=0)
  y_train_b = np.concatenate((y_t_b,y_b),axis=0)

  mnb = multinomial_nb(x_train_a, y_train_a, x_train_b, y_train_b).reshape(1200, 1)
  gnb = gaussian_nb(x_train_a, y_train_a, x_train_b, y_train_b).reshape(1200, 1)
  lr = logistic_regression(x_train_a, y_train_a, x_train_b, y_train_b).reshape(1200, 1)
  svm = linear_svc(x_train_a, y_train_a, x_train_b, y_train_b).reshape(1200, 1)
  knn = k_nearest_neighbors(x_train_a, y_train_a, x_train_b, y_train_b).reshape(1200, 1)
  dt = decision_tree(x_train_a, y_train_a, x_train_b, y_train_b).reshape(1200, 1)
  x = np.concatenate((mnb,  lr, svm, dt), axis=1)
  clf = DecisionTreeClassifier(random_state=0)
  clf.fit(x, y_train_b)


  dt = decision_tree(x_train_a, y_train_a, x_test, y_test).reshape(600, 1)
  mnb = multinomial_nb(x_train_a, y_train_a, x_test, y_test).reshape(600, 1)
  gnb = gaussian_nb(x_train_a, y_train_a, x_test, y_test).reshape(600, 1)
  lr = logistic_regression(x_train_a, y_train_a, x_test, y_test).reshape(600, 1)
  svm = linear_svc(x_train_a, y_train_a, x_test, y_test).reshape(600, 1)
  knn = k_nearest_neighbors(x_train_a, y_train_a, x_test, y_test).reshape(600, 1)
  x_test_new = np.concatenate(( mnb,  lr,svm, dt), axis=1)
  print(x_test_new[0:10])
  pred = clf.predict(x_test_new)
  print(pred[0:10])
  return precision_recall_fscore_support(y_test, pred, average = 'binary')

def multinomial_nb(x_train, y_train, x_test, y_test):
  mnb = MultinomialNB()
  mnb.fit(x_train, y_train)
  return mnb.predict(x_test)
  # return precision_recall_fscore_support(y_test, pred, average = 'binary')
  # return mnb.score(x_test, y_test)
  # fpr, tpr, thresholds = roc_curve(y_test, pred)
  # roc_auc = auc(y_test, pred)
  # plt.plot(fpr, tpr, label='roc curve')
  # plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
  # plt.xlim([0.0, 1.0])
  # plt.ylim([0.0, 1.0])
  # plt.xlabel('FPR')
  # plt.ylabel('TPR')
  # plt.show()

def gaussian_nb(x_train, y_train, x_test, y_test):
  gnb = GaussianNB()
  gnb.fit(x_train, y_train)
  return gnb.predict(x_test)
  # return precision_recall_fscore_support(y_test, pred, average = 'binary')
  fpr, tpr, thresholds = roc_curve(y_test, pred)
  # roc_auc = auc(y_test, pred)
  plt.plot(fpr, tpr, label='roc curve')
  plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('FPR')
  plt.ylabel('TPR')
  plt.show()
# return gnb.score(x_test, y_test)

def linear_svc(x_train, y_train, x_test, y_test):
  lsvc = LinearSVC(random_state=0)
  lsvc.fit(x_train, y_train)
  return lsvc.predict(x_test)
  # return precision_recall_fscore_support(y_test, pred, average = 'binary')
  fpr, tpr, thresholds = roc_curve(y_test, pred)
  # roc_auc = auc(y_test, pred)
  plt.plot(fpr, tpr, label='roc curve')
  plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('FPR')
  plt.ylabel('TPR')
  plt.show()
  # return cross_val_score(lsvc, x, y, cv = 10, scoring = 'accuracy')

def k_nearest_neighbors(x_train, y_train, x_test, y_test):
  knn = KNeighborsClassifier(n_neighbors=3)
  knn.fit(x_train, y_train)
  # return knn.score(x_test, y_test)
  return knn.predict(x_test)
  # return precision_recall_fscore_support(y_test, pred, average = 'binary')
  # print(pred)
  fpr, tpr, thresholds = roc_curve(y_test, pred)
  # roc_auc = auc(y_test, pred)
  plt.plot(fpr, tpr, label='roc curve')
  plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('FPR')
  plt.ylabel('TPR')
  plt.show()

def logistic_regression(x_train, y_train, x_test, y_test):
  lr = LogisticRegression()
  lr.fit(x_train, y_train)
  return lr.predict(x_test)
  # return precision_recall_fscore_support(y_test, pred, average = 'binary')
  # print(pred)
  fpr, tpr, thresholds = roc_curve(y_test, pred)
  # roc_auc = auc(y_test, pred)
  plt.plot(fpr, tpr, label='roc curve')
  plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('FPR')
  plt.ylabel('TPR')
  plt.show()

if __name__ == '__main__':
  x_train = pd.read_csv(filepath_or_buffer='train_bag_of_words_5.csv', header=None, sep=',').values
  y_train = pd.read_csv(filepath_or_buffer='train_classes_5.txt', header=None, sep=',').values.ravel()
  x_test = pd.read_csv(filepath_or_buffer='test_bag_of_words_0.csv', header=None, sep=',').values
  y_test = pd.read_csv(filepath_or_buffer='test_classes_0.txt', header=None, sep=',').values.ravel()
  
  # print(x_train_a.shape)
  # print(x_train_b.shape)
  # x = np.concatenate((x_train,x_test), axis=0)
  # y = np.concatenate((y_train,y_test), axis=0).ravel()
  print("read in data")

  # print(linear_svc(x,y))
  # cross_validation(x_train, y_train, x_test, y_test)
  # print(logistic_regression(x_train,y_train,x_test,y_test))
  # print(k_nearest_neighbors(x_train, y_train, x_test, y_test))
  # print(multinomial_nb(x_train, y_train, x_test, y_test))
  # print(decision_tree(x_train, y_train, x_test, y_test))
  print(ensemble_decision_tree(x_train,y_train,x_test,y_test))

  # print(gaussian_nb(x_train, y_train, x_test, y_test))
  # print(linear_svc(x_train, y_train, x_test, y_test))
