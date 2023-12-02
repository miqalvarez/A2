# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import crossk_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Read csv file
test = pd.read_csv('A2-bank/test.csv', header=None, skiprows=1, sep=';')
train = pd.read_csv('A2-bank/train.csv', header=None, skiprows=1, sep=';')

# Split data into X and y
X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]
X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

print('X_train:', X_train.shape)
print('y_train:', y_train.shape)


print('y_train data type:', y_train.dtypes)
print('Unique labels in y_train:', y_train.unique())


# Apply cross validation using 10 to find best C
C = np.arange(0.1, 2.1, 0.1)
scores = []
for c in C:
    clf = svm.SVC(kernel='linear', C=c)
    score = cross_val_score(clf, X_train, y_train, cv=10)
    scores.append(score.mean())

# Plot the graph
plt.plot(C, scores)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Cross Validation')
plt.show()

# Find the best C
best_C = C[np.argmax(scores)]
print('Best C:', best_C)

# Train the model using best C
clf = svm.SVC(kernel='linear', C=best_C)
clf.fit(X_train, y_train)

# Predict the test data
y_pred = clf.predict(X_test)

# Report the expected classification error obtained from cross-validation and compare it with the classification error on the test set
print('Expected classification error:', 1 - np.max(scores))
print('Classification error on the test set:', 1 - clf.score(X_test, y_test))

