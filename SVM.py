import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv('A2-bank/full.csv', sep=';')
X = df.drop('y', axis=1)
y = df['y']

# Split the dataset into training and test sets with a 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Print 0 and 1 counts in y_train
print('y_train 0 count: ', np.count_nonzero(y_train == 0))
print('y_train 1 count: ', np.count_nonzero(y_train == 1))

'''
# Load the training dataset from a txt using \t as separator
df_train = pd.read_csv('A2-ring/A2-ring-separable.txt', sep='\t')
X_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1]

# Load the test dataset
df_test = pd.read_csv('A2-ring/A2-ring-test.txt', sep='\t')
X_test = df_test.iloc[:, :-1]
y_test = df_test.iloc[:, -1]
'''

# Define the parameter grid for GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.01, 0.1, 1, 10],
              'kernel': ['rbf', 'poly', 'sigmoid'],
              'max_iter': [-1]}

# Create the SVM classifier with a different kernel
svm = SVC(kernel='rbf')

# Perform grid search with cross-validation to find the best parameters
grid_search = GridSearchCV(svm, param_grid, cv=10)
grid_search.fit(X_train, y_train)

# Get the best parameters and 
best_params = grid_search.best_params_

# Create a new SVM classifier with the best parameters
svm = SVC(C=best_params['C'], kernel='rbf')

# Fit the classifier to the training data
svm.fit(X_train, y_train)

# Predict the test data
y_pred = svm.predict(X_test)

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Compute and plot error rate
error_rate = np.mean(y_pred != y_test)
print('Error rate: ', error_rate)

# Plot PCA of predicted data
from sklearn.decomposition import PCA
pca = PCA(n_components=18)
X_pca = pca.fit_transform(X_test)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred)
plt.title('PCA of predicted data')
plt.show()

