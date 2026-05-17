import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold


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


def classification_metrics(y_true, y_pred):
    """
    Calculates common classification metrics: accuracy, precision, 
    recall and F1-score.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_model(y_true, y_pred, n_features, model_name=None):
    """
    Evaluate the model using both classification metrics and scoring functions
    """
    metrics = classification_metrics(y_true, y_pred)

    metrics.update({
        "profit": profit_function(y_true, y_pred),
        "score": scoring_function(y_true, y_pred, n_features),
        "n_features": n_features,
        "model": model_name
    })

    return metrics


def cross_val_eval(model, X, y, n_features, model_name):
    """
    Evaluate the model using cross-validation and return summary metrics.
    """

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []

    for train_idx, val_idx in cv.split(X, y):

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        metrics = evaluate_model(y_val, preds, n_features, model_name)
        fold_results.append(metrics)

    df = pd.DataFrame(fold_results)

    summary = df.mean(numeric_only=True).to_dict()

    summary["accuracy_std"] = df["accuracy"].std()
    summary["precision_std"] = df["precision"].std()
    summary["recall_std"] = df["recall"].std()
    summary["f1_std"] = df["f1"].std()
    summary["profit_std"] = df["profit"].std()
    summary["score_std"] = df["score"].std()


    summary["model"] = model_name
    summary["n_features"] = n_features

    return summary