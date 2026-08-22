# Gradient Descent for Linear Regression
# w = w - learning_rate * dw
# b = b - learning_rate * db
# dw = (1/m) * np.dot(X.T, (y_pred - y))
# db = (1/m) * np.sum(y_pred - y)
# Steps:
# Training loop:
# - Initialize parameters w and b to zero or small random values.
# - For a specified number of iterations (epochs):
#   - Compute the predictions y_pred.
#   - Compute the gradients dw and db.
#   - Update the parameters w and b.

from typing import List
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets
# import matplotlib
# matplotlib.use('Agg')  # Use a GUI backend
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)  # Init weights
        self.b = 0
        print_interval = max(1, self.n_iters // 10)

        # Gradient descent
        for iteration in range(self.n_iters):
            y_predicted = np.dot(X, self.w) + self.b
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if (iteration + 1) % print_interval == 0:
                error = np.mean((y - (np.dot(X, self.w) + self.b)) ** 2)
                print(f"Epoch {iteration + 1}: MSE = {error:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        linear_model = np.dot(X, self.w) + self.b
        return linear_model


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


# Example usage
if __name__ == "__main__":
    # Generate a simple dataset
    X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=30, random_state=42)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train, color='blue', label='Training data')
    plt.scatter(X_test, y_test, color='red', label='Testing data')
    plt.xlabel('Feature')   
    plt.ylabel('Target')
    plt.title('Linear Regression Dataset')
    plt.legend()
    plt.show()

    # Create and train the linear regression model
    model = LinearRegression(learning_rate=0.01, n_iters=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    mse_value = mse(y_test, y_pred)
    print(f'Mean Squared Error on test set: {mse_value}')   

    y_pred_line = model.predict(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, y_pred_line, color='red', label='Regression line')
    plt.xlabel('Feature')  
    plt.ylabel('Target')
    plt.title('Linear Regression Fit')
    plt.legend()   
    plt.show()
