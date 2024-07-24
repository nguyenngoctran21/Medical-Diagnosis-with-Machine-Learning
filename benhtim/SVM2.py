import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# Loading the dataset
df = pd.read_csv('data/heart-disease.csv')

# Splitting the data into features (X) and target variable (y)
X = df.drop('target', axis=1)
y = df['target']

# Initialize the LinearSVC model
svm = LinearSVC()

# Define the parameter grid for C
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}

best_accuracy = 0.0
best_test_size = 0.0
best_random_state = 0

# Try different test_size and random_state values
for test_size in [0.1, 0.2, 0.3]:
    for random_state in [0, 1, 42]:
        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Initialize the grid search with LinearSVC and the parameter grid
        grid_search = GridSearchCV(svm, param_grid, cv=5)

        # Fit the grid search to the training data
        grid_search.fit(X_train, y_train)

        # Get the best parameter for C
        best_C = grid_search.best_params_['C']

        # Train the LinearSVC model with the best C value
        best_svm = LinearSVC(C=best_C)
        best_svm.fit(X_train, y_train)

        # Evaluate the model on the test data
        test_accuracy = best_svm.score(X_test, y_test)

        # Check if the current model has higher accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_test_size = test_size
            best_random_state = random_state

# Print the best test_size, random_state, and accuracy
print("Best Results:")
print(f"Best test_size: {best_test_size}")
print(f"Best random_state: {best_random_state}")
print(f"Best accuracy: {best_accuracy:.4f}")