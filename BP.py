import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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

X_train, y_train, X_test, y_test = load_dataset('star')
n_classes = len(np.unique(y_train))

# Define the Keras model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

# Predict the test data
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test.iloc[:, i], y_pred.iloc[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# Plot ROC curve
plt.figure()
lw = 2
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='lower right')
plt.show()


# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()

# Plot PCA of predicted data
pca = PCA(n_components=64)
X_pca = pca.fit_transform(X_test)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred)
plt.title('PCA of predicted data')
plt.show()

# Compute classification error percentage
error_rate = 100 * (cm[0, 1] + cm[1, 0]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
print('Error rate: ', error_rate)








