import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

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
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Split the dataset into training and test sets with a 80:20 ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

    elif n == 'star':
        # Load the training dataset from a txt using \t as separator
        df_train = pd.read_csv('Star\star.csv', sep=',')
        X = df_train.iloc[:, :-1]
        y = df_train.iloc[:, -1]
        
        # Split the dataset into training and test sets with a 80:20 ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_dataset('ring')



print(f"X_train type: {type(X_train)}, shape: {X_train.shape}")
print(f"y_train type: {type(y_train)}, shape: {y_train.shape}")
n_classes = len(np.unique(y_train))

# Define the Keras model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')

# Train the model
history = model.fit(X_train, y_train, epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Predict the test data
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

if n_classes == 2:
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    
else:
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    y_pred_bin = label_binarize(y_pred, classes=np.unique(y_test))

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for each class
plt.figure(figsize=(10, 8))
lw = 2
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
if n_classes == 2:
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
else:
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve (area = {:.2f})'.format(roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Multiclass')
plt.legend(loc='lower right')
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







