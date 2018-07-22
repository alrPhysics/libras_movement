# Classify [LIBRAS](http://archive.ics.uci.edu/ml/datasets/Libras+Movement) movement based on preprocessed video data.

The raw dataset consists of 90 features. These were reduced to 4, 6, and 10 dimensions using PCA in order to investigate the performance of various ML classifiers based on the number of PCA dimensions used. Results can be seen in the Jupyter notebook. The classifiers are optimized with GridSearchCV using the data that is reduced to 6 and 10 dimensions.

### ML Classifiers Used:
* LogisticRegression
* MLPClassifier
* DecisionTreeClassifier
* GaussianProcessClassifier
* KNeighborsClassifier

### Additional Notes:
* May continue playing with dimensionality reduction
* May investigate KNeighbors further
