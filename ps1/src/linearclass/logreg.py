import numpy as np
import math
import util

def main(train_path, valid_path, save_path, plot_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    np.savetxt(save_path, clf.predict(x_val))
    util.plot(x_val, y_val, clf.theta, plot_path)
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=1.0, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def h(self, x):
        z = np.dot(x, self.theta)
        return 1 / (1 + np.exp(-z))

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (N, D).
            y: Training example labels. Shape (N,).
        """
        # *** START CODE HERE ***
        N, D = x.shape
        self.theta = np.zeros(D)
        epoch = 0

        while epoch < self.max_iter:
            grad_j = (-1/N) * np.dot(x.T, y - self.h(x))
            hessian = (1/N) * np.dot(x.T, x) * np.dot((1 - self.h(x)).T, self.h(x))
            next_theta = self.theta - self.step_size * np.dot(np.linalg.inv(hessian), grad_j)
            if np.linalg.norm(self.theta - next_theta, ord=1) < self.eps:
                break
            self.theta = next_theta
            epoch += 1

        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-np.dot(x, self.theta)))
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt',
         plot_path="./logreg_plot_1.png")

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt',
         plot_path="./logreg_plot.png")
