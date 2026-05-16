def scoring_function(y_true, y_pred, n_features):
    """
    Scoring function for evaluating the performance
    of a binary classification model.
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()

    score = tp * 10 - fp * 5 - n_features * 200
    return score


def profit_function(y_true, y_pred):
    """
    Scoring function for evaluating profit withou feature cost.
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()

    score = tp * 10 - fp * 5
    return score