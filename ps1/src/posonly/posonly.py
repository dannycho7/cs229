import numpy as np
import util
import sys

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    output_path_true = save_path.replace(WILDCARD, 'true')
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_adjusted = save_path.replace(WILDCARD, 'adjusted')

    # *** START CODE HERE ***
    # Part (a): Train and test on true labels
    x_train_t, y_train_t = util.load_dataset(train_path, label_col='t', add_intercept=True)
    clf = LogisticRegression()
    clf.fit(x_train_t, y_train_t)
    
    x_test_t, y_test_t = util.load_dataset(test_path, label_col='t', add_intercept=True)
    util.plot(x_test_t, y_test_t, clf.theta, "./plot_a.png")
    np.savetxt(output_path_true, clf.predict(x_test_t))
    # Make sure to save predicted probabilities to output_path_true using np.savetxt()
    # Part (b): Train on y-labels and test on true labels
    x_train_y, y_train_y = util.load_dataset(train_path, label_col='y', add_intercept=True)
    clf = LogisticRegression()
    clf.fit(x_train_y, y_train_y)
    
    util.plot(x_test_t, y_test_t, clf.theta, "./plot_b.png")
    np.savetxt(output_path_naive, clf.predict(x_test_t))
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    # Part (f): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to output_path_adjusted
    x_val_y, y_val_y = util.load_dataset(train_path, label_col='y', add_intercept=True)
    clf = LogisticRegression()
    clf.fit(x_val_y, y_val_y)
    alpha = np.sum(clf.predict(x_val_y) * y_val_y) / np.sum(y_val_y)

    util.plot(x_test_t, y_test_t, clf.theta, "./plot_f.png", correction=alpha)
    np.savetxt(output_path_adjusted, clf.predict(x_test_t) / alpha)
    # *** END CODER HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')
