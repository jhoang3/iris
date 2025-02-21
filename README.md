# 2nd project to learn about classification - Iris Flowers

**Project Description**
This project involves analyzing a dataset on species of Iris flowers. The goal of this project is to train accurate machine learning models to best classify what a given flower is based on its sepal and petal length/width.

After visually analyzing the pairplot and univariate statistics with the provided features, we see that the sestosa species typically has the shortest sepal length, petal length and width, followed by the versicolor species, then virginica.

To get an understanding of the usage of ML models, I've depicted a support vector machine based on 2 features, sepal length and width as arbitary features. 

Afterwards, a variety of ML models will be trained and tested with the entire set of features. Each model will be evaluated, tuned, then evaluated again to get an understanding of what changed and how exactly that impacted the metrics that will be later interpreted.

Interpreting the values, based on a single test set. We find that several models perform similarly. We will choose to pick a "best" model on accuracy. This is because there is no particular emphasise on minimizing false negatives (otherwise pick recall) or false positives (otherwise pick precision). Deciding on F1 scores may be acceptable, but since our dataset is balanced and all classes are equivalently weighted, we will base it on accuracy.

**Results**
| Classification Model      |   Accuracy Train (%) |   Accuracy Test (%) |
|:--------------------------|---------------:|--------------:|
| Decision Tree tuned       |       98.33  |      96.67 |
| Random Forest tuned       |       96.67  |      96.67 |
| Naive Bayes               |       95.83 |      96.67 |
| Naive Bayes tuned         |       95.83 |      96.67 |
| Neural Network tuned      |       98.33 |      96.67 |

Based off of our results, we've decided on using a tuned random forest as our model to classify Iris flower species. Although it does not have the best results compared to a decision tree or neural network, a tuned random forest was decided upon because its structure is more explainable than a neural network and it's more applicable to unseen data compared to a decision tree. 