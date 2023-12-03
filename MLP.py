import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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
n_classes = len(np.unique(y_train))

# Define the parameter grid for GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(10,30,10),(20,), (50,), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
# Create the BP classifier
bp = MLPClassifier(max_iter=1)

# Perform grid search with cross-validation to find the best parameters
grid_search = GridSearchCV(bp, param_grid, cv=3, n_jobs=-1)
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

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
y_test = pd.get_dummies(y_test)
y_pred = pd.get_dummies(y_pred)

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test.iloc[:, i], y_pred.iloc[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot the ROC curve
plt.figure()
lw = 2
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of class {0} (area = {1:0.3f})'.format(i, roc_auc[i]))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

# Plot PCA of predicted data
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_test)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred)
plt.title('PCA of predicted data')
plt.show()

# Compute classification error percentage
error_rate = 100 * (cm[0, 1] + cm[1, 0]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
print('Error rate: ', error_rate)

# Plot the training and validation loss
plt.plot(bp.loss_curve_)
plt.title('Training and validation loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show() 



