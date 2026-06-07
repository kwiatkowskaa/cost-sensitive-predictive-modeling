import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


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
    Scoring function for evaluating profit without feature cost.
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()

    score = tp * 10 - fp * 5
    return score

def calculate_profit(y_true, probs, max_customers=1000):

    ranking = np.argsort(probs)[::-1]
    y_sorted = np.array(y_true)[ranking]

    best_profit = -np.inf
    best_n = 0

    tp = 0
    fp = 0

    upper_limit = min(max_customers, len(y_sorted))

    for n in range(1, upper_limit + 1):

        if y_sorted[n - 1] == 1:
            tp += 1
        else:
            fp += 1

        profit = tp * 10 - fp * 5

        if profit > best_profit:
            best_profit = profit
            best_n = n

    return best_profit, best_n


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


def cross_val_eval(model, X, y, n_features, model_name, threshold=None, top_n=1000):
    """
    Evaluate the model using cross-validation and return summary metrics.
    """

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []

    for train_idx, val_idx in cv.split(X, y):

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)

        scores = get_prediction_scores(model, X_val)
        
        if top_n is not None:
            # preds = top_n_predictions(scores, top_n) # old version
            effective_top_n = int(top_n * len(X_val) / len(X))
            preds = top_n_predictions(scores, effective_top_n)

        elif threshold is not None:
            preds = (scores >= threshold).astype(int)
        
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
    summary["threshold"] = threshold
    summary["top_n"] = top_n

    return summary


def get_prediction_scores(model, X):

    # probability models
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]

    # margin-based models
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)

    # fallback
    else:
        scores = model.predict(X)

    return scores


def top_n_predictions(scores, top_n):
    """
    Select top_n observations with highest scores.
    """

    preds = np.zeros(len(scores), dtype=int)

    top_idx = np.argsort(scores)[::-1][:top_n]

    preds[top_idx] = 1

    return preds
