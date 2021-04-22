import numpy as np
import util


def main(train_path, valid_path, save_path, plot_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    clf = GDA()
    clf.fit(x_train, y_train)
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    np.savetxt(save_path, clf.predict(x_val))
    util.plot(x_val, y_val, clf.theta, plot_path)
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=1.0, max_iter=10000, eps=1e-5,
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

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (N, D).
            y: Training example labels. Shape (N,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        # Write theta in terms of the parameters
        N, D = x.shape
        self.theta = np.zeros(D + 1)

        phi = (1/N) * np.sum(y == 1)
        mu_0 = (y == 0).dot(x) / (np.sum(y == 0))
        mu_1 = (y == 1).dot(x) / (np.sum(y == 1))
        mu_x = np.array([mu_0, mu_1])[y.astype(int)]
        sigma = (1/N) * np.sum((x - mu_x).dot((x - mu_x).T))

        self.theta[0] = -np.log((1 - phi)/phi) + (1/2) * (mu_0.dot(mu_0.T) - mu_1.dot(mu_1.T)) / sigma
        self.theta[1:] = (mu_1 - mu_0) / sigma
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (N, D).

        Returns:
            Outputs of shape (N,).
        """
        # *** START CODE HERE ***
        print(np.dot(x, self.theta))
        return 1 / (1 + np.exp(-np.dot(x, self.theta)))
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt',
         plot_path="./gda_plot_1.png")

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt',
         plot_path="./gda_plot_2.png")
