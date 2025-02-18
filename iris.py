import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn as sk
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay




plt.style.use('ggplot')

path = kagglehub.dataset_download("arshid/iris-flower-dataset")

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "arshid/iris-flower-dataset/versions/1",
  "Iris.csv",
)

# print(df.head()) # 5 rows of data, we have length and width for speal and pedal + the species
# print(df.describe()) # no striking outliers or anomalies (e.g negative numbers)
# print(df.info()) # all columns are float64
# print(df.isnull().sum()) # no null values
# print(df.duplicated().sum()) # no duplicates

# ax = sns.pairplot(df, hue='species') # visualizing the data
# ax.fig.suptitle("Pairplot of Iris Dataset", y=1.02)
# plt.show()

x = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']
# first model to use: suport vector machines
  # is effective in high dimensional spaces
'''
Empirical studies show that the best results are obtained 
if we use 20-30% of the data for testing, 
and the remaining 70-80% of the data for training.
'''
# pick 2 columns of x to visualize
x2D = x.iloc[:, :2]
# x.iloc[:, :2] is saying select all rows, up to but not including col index 2, e.g 0 & 1
x2_train, x2_test, y2_train, y2_test = train_test_split(x2D, y, test_size=0.2)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
                                  # takes x and y input columns, test_size is 20%, can pass
                                  # random_state=int for consistent train/test datasets
          # Define different kernel functions

kernel_functions = ['linear', 'poly', 'rbf', 'sigmoid']
# a list of kernel functions to cross validate to decide which is best

scaler = sk.preprocessing.StandardScaler()
# standardize data
x_train = scaler.fit_transform(x_train)
# fit_transform is typically done on training data
# fit_transform: "Learns transformation parameters from the data & applies the transformation to the same data."
x_test = scaler.transform(x_test)
# transform(): uses previously fit data on the new set
# use on test set to compare
x2_train = scaler.fit_transform(x2_train)
x2_test = scaler.transform(x2_test)                                

                                
# we will do 2d analysis first
for i, kernel in enumerate(kernel_functions):
  clf = svm.SVC(kernel=kernel)
  # clf represents classifier
  
  clf.fit(x2_train, y2_train)
  # trains the classifier
  acc = clf.score(x2_test, y2_test)
  # calculates the accuracy based on the remaining testing data
  print(f'{kernel} {acc}')
  
  # plt.subplot(2,2,i+1)
  # makes a plot with 2x2 dimensions, we are modifying the (i+1)-th subplot
  y_pred = clf.predict(x2_test)
  # accuracy = clf.score(x2_test, y2_test)
  # accuracy_score() can be used similarly for weighted acc, only for classification
  # can be customized etc.

  # Plot decision boundary and data points
  plt.subplot(2, 2, i+1)
  # sns.scatterplot(x=x2_train[:, 0], y=x2_train[:, 1], hue=y2_train, palette='Set2')
  # map y2_train to a color palette
  colors = {'Iris-setosa': 1, 'Iris-versicolor':2, 'Iris-virginica':3}
  

  # Put the result into a color plot
  disp = DecisionBoundaryDisplay.from_estimator(clf, x2_train, response_method='predict', alpha=0.8)
  # colors in the decision boundary
  disp.ax_.scatter(x2_train[:, 0], x2_train[:, 1], c=y2_train.map(colors), cmap=plt.cm.coolwarm, edgecolor='k')
  # map the colors to the training data and plots the points
  plt.show()
  
  

# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.tight_layout()
# plt.show()
  