import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def load_dataset(n):
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    if n == 'ring':
        # Load the training dataset from a txt using \t as separator
        df_train = pd.read_csv('A2-ring/A2-ring-separable.txt', sep='\t')
        X_train = df_train.iloc[:, :-1]
        y_train = df_train.iloc[:, -1]

        # Load the test dataset
        df_test = pd.read_csv('A2-ring/A2-ring-test.txt', sep='\t')
        X_test = df_test.iloc[:, :-1]
        y_test = df_test.iloc[:, -1]
    elif n == 'bank':
        # Load the dataset
        df = pd.read_csv('A2-bank/full.csv', sep=';')
        X = df.drop('y', axis=1)
        y = df['y']

        # Split the dataset into training and test sets with a 80:20 ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    elif n == 'star':
        # Load the training dataset from a txt using \t as separator
        df_train = pd.read_csv('Star\star.csv', sep=',')
        X = df_train.iloc[:, :-1]
        y = df_train.iloc[:, -1]
        
        # Split the dataset into training and test sets with a 80:20 ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_dataset('ring')

# Define the parameter grid for GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100],
              'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

# Create the SVM classifier with a different kernel
svm = SVC()

# Perform grid search with cross-validation to find the best parameters
grid_search = GridSearchCV(svm, param_grid, cv=10, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and 
best_params = grid_search.best_params_
print(best_params)

# Create a new SVM classifier with the best parameters
svm = SVC(C=best_params['C'], kernel=best_params['kernel'], gamma='scale')

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

# Plot PCA of predicted data
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred)
plt.title('PCA of predicted data')
plt.show()

# For binary classification, handle the case where the confusion matrix is a 2x2 matrix
if cm.shape == (2, 2):
    # Compute classification error percentage
    error_rate = 100 * (cm[0, 1] + cm[1, 0]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])

elif cm.shape == (3, 3):
    # Compute classification error percentage
    error_rate = 100 * (cm[0, 1] + cm[0, 2] + cm[1, 0] + cm[1, 2] + cm[2, 0] + cm[2, 1]) / (
                cm[0, 0] + cm[0, 1] + cm[0, 2] + cm[1, 0] + cm[1, 1] + cm[1, 2] + cm[2, 0] + cm[2, 1] + cm[2, 2])
    

print('Error rate: ', error_rate)
