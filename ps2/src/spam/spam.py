import svm
import math
import util
import logreg
import collections
import numpy as np

def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces, i.e. use split(' ').
    For normalization, you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """
    # *** START CODE HERE ***
    return list(map(str.lower, message.split(' ')))
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """
    # *** START CODE HERE ***
    wordToCount = {}
    for message in messages:
        words = list(dict.fromkeys(get_words(message)))
        for word in words:
            wordToCount[word] = wordToCount.get(word, 0) + 1
    word_dictionary = {}
    for word, count in wordToCount.items():
        if count >= 5:
            word_dictionary[word] = len(word_dictionary)
    return word_dictionary
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """    
    # *** START CODE HERE ***
    I, J = len(messages), len(word_dictionary)
    transformed = np.zeros((I, J))
    for i, message in enumerate(messages):
        for word in get_words(message):
            if word in word_dictionary:
                transformed[i, word_dictionary[word]] += 1
    return transformed
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of the fitted model, consisting of the 
    learned model parameters.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """
    # *** START CODE HERE ***
    N, D = matrix.shape
    num_words = matrix.sum(axis=1)
    phi_y0 = (1 + np.sum(matrix * (labels == 0)[:, np.newaxis], axis=0)) / (D + (labels == 0).dot(num_words))
    phi_y1 = (1 + np.sum(matrix * (labels == 1)[:, np.newaxis], axis=0)) / (D + (labels == 1).dot(num_words))
    phi_y = (labels == 1).mean()
    return (phi_y0, phi_y1, phi_y)
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model (int array of 0 or 1 values)
    """
    # *** START CODE HERE ***
    phi_y0, phi_y1, phi_y = model
    # Note: these are log probabilites. we don't need to compute p(y|x) as p(x|y=i) works for relative comparison.
    p_xj_y0 = (matrix * phi_y0)
    p_xj_y0[p_xj_y0 == 0] = 1
    p_y0_x = np.log(p_xj_y0).sum(axis=1) + np.log(1 - phi_y)
    p_xj_y1 = (matrix * phi_y1)
    p_xj_y1[p_xj_y1 == 0] = 1
    p_y1_x = np.log(p_xj_y1).sum(axis=1) + np.log(phi_y)

    prediction = (p_y1_x > p_y0_x).astype(int)
    return prediction
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    phi_y0, phi_y1, phi_y = model
    # Note: these are log probabilites.
    indicativeness = np.log(phi_y1) - np.log(phi_y0)
    word_idxs = indicativeness.argsort()[::-1][:5]
    words = []
    for word_idx in word_idxs:
        for k, v in dictionary.items():
            if v == word_idx:
                words.append(k)
                continue
    return words
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use validation set accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spam or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    best_acc = None
    best_radius = None
    for radius in radius_to_consider:
        prediction = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracy = (val_labels == prediction).mean()
        if best_acc is None or best_acc < accuracy:
            best_radius = radius
            best_acc = accuracy
    return best_radius
    # *** END CODE HERE ***

    
def compute_best_logreg_learning_rate(train_matrix, train_labels, val_matrix, val_labels, learning_rates_to_consider):
    """Compute the best logistic regression learning rate using the provided training and evaluation datasets.

    You should only consider learning rates within the learning_rates_to_consider list.
    You should use validation set accuracy as a metric for comparing the different learning rates.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spam or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        learning_rates_to_consider: The learning rates to consider

    Returns:
        The best logistic regression learning rate which maximizes validation set accuracy.
    """
    # *** START CODE HERE ***
    best_acc = None
    best_lr = None
    for lr in learning_rates_to_consider:
        prediction = logreg.train_and_predict_logreg(train_matrix, train_labels, val_matrix, lr)
        accuracy = (val_labels == prediction).mean()
        if best_acc is None or best_acc < accuracy:
            best_lr = lr
            best_acc = accuracy
    return best_lr
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)


    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


    train_matrix = util.load_bert_encoding('bert_train_matrix.tsv.bz2')
    val_matrix = util.load_bert_encoding('bert_val_matrix.tsv.bz2')
    test_matrix = util.load_bert_encoding('bert_test_matrix.tsv.bz2')
    
    best_learning_rate = compute_best_logreg_learning_rate(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.001, 0.0001, 0.00001, 0.000001])

    print('The best learning rate for logistic regression is {}'.format(best_learning_rate))

    logreg_predictions = logreg.train_and_predict_logreg(train_matrix, train_labels, test_matrix, best_learning_rate)

    logreg_accuracy = np.mean(logreg_predictions == test_labels)

    print('The Logistic Regression model with BERT encodings had an accuracy of {} on the testing set'.format(logreg_accuracy))
    

if __name__ == "__main__":
    main()
