import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold

# Load the dataset
file_path = 'Star\star.csv'
df = pd.read_csv(file_path)

# Split the dataset into features and labels
X = df.iloc[:, 1:-2].values
y = df['class'].values

# One-hot encode the labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the dataset into training and test sets with a 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(np.unique(y)), activation='softmax'))  # Capa de salida con activación softmax para clasificación multiclase

# Compile the model using cross-validation of 10
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for train_index, val_index in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=32, validation_data=(X_val_fold, y_val_fold))

# Evaluate the model
y_pred = np.argmax(model.predict(X_test), axis=1)

# Plot the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.matshow(conf_mat)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# Calculate the error rate
error_rate = 1 - np.sum(np.diag(conf_mat)) / np.sum(conf_mat)
print(f'Error Rate: {error_rate}')


# Plot the PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded)
plt.title('PCA')
plt.show()
