import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
'''
# Load the dataset
df = pd.read_csv('A2-bank/full.csv', sep=';')
X = df.drop('y', axis=1)
y = df['y']

# Split the dataset into training and test sets with a 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
'''

# Load the training dataset from a txt using \t as separator
df_train = pd.read_csv('A2-ring/A2-ring-separable.txt', sep='\t')
X_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1]

# Load the test dataset
df_test = pd.read_csv('A2-ring/A2-ring-test.txt', sep='\t')
X_test = df_test.iloc[:, :-1]
y_test = df_test.iloc[:, -1]


# Print 0 and 1 counts in y_train
print('y_train 0 count: ', np.count_nonzero(y_train == 0))
print('y_train 1 count: ', np.count_nonzero(y_train == 1))

# Define the parameter grid for GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(100,), (50, 50), (50, 30, 20)],
    'activation': ['relu', 'tanh', 'logistic'],
    'alpha': [0.0001, 0.001, 0.01],
}

# Create the BP classifier
bp = MLPClassifier(max_iter=1000)

# Perform grid search with cross-validation to find the best parameters
grid_search = GridSearchCV(bp, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(best_params)

# Create a new BP classifier with the best parameters
bp = MLPClassifier(hidden_layer_sizes=best_params['hidden_layer_sizes'], alpha=best_params['alpha'])

# Fit the classifier to the training data
bp.fit(X_train, y_train)

# Predict the test data
y_pred = bp.predict(X_test)

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
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred)
plt.title('PCA of predicted data')
plt.show()
