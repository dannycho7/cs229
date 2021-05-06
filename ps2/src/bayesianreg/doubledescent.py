import numpy as np
import util

# Noise ~ N(0, sigma^2)
sigma = 0.5
# Dimension of x
d = 500
# Theta ~ N(0, eta^2*I)
eta = 1/np.sqrt(d)
# Scaling for lambda to plot
scale_list = [0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4]
# List of dataset sizes
n_list = [250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]


def ridge_regression(train_path, validation_path):
    """
    For a specific training set, obtain theta_hat under different l2 regularization strengths
    and return validation error.

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.

    Return:
        val_err: List of validation errors for different scaling factors of lambda in scale_list.
    """
    # *** START CODE HERE ***
    train_xs, train_ys = util.load_dataset(train_path)
    val_xs, val_ys = util.load_dataset(validation_path)
    val_errs = []
    Nv, = val_ys.shape
    l_opt = np.identity(d) / (2 * eta**2)
    for scale in scale_list:
        reg = scale * l_opt
        theta = np.linalg.pinv(2 * reg * sigma ** 2 + train_xs.T.dot(train_xs)).dot(train_xs.T.dot(train_ys))
        mse_theta = (np.linalg.norm(val_xs.dot(theta) - val_ys)**2) / Nv
        val_errs.append(mse_theta)
    return val_errs
    # *** END CODE HERE


if __name__ == '__main__':
    val_err = []
    for n in n_list:
        val_err.append(ridge_regression(train_path='train%d.csv' % n, validation_path='validation.csv'))
    val_err = np.asarray(val_err).T
    util.plot(val_err, 'doubledescent.png', n_list, scale_list)
