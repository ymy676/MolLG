import numpy as np
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score
)


def _to_numpy(x):
    """
    Convert torch tensor or list to numpy array.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    return np.array(x)


def mae(y_true, y_pred):
    """
    Mean Absolute Error
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    return mean_absolute_error(y_true, y_pred)


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    return np.sqrt(mean_squared_error(y_true, y_pred))


def r2(y_true, y_pred):
    """
    R^2 Score
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    return r2_score(y_true, y_pred)

def roc_auc(y_true, y_pred):
    """
    Compute ROC AUC score for binary classification.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    return roc_auc_score(y_true, y_pred)

def regression_metrics(y_true, y_pred):
    """
    Return all regression metrics in a dict.
    """
    return {
        "RMSE": rmse(y_true, y_pred),
    }

def binary_classification_metrics(y_true, y_pred):
    """
    Return all binary classification metrics in a dict.
    """
    return {
        "ROC-AUC": roc_auc(y_true, y_pred)
    }

def multitask_binary_classification_metrics(y_true, y_pred):
    """
    Macro-average ROC-AUC for multi-task binary classification.
    """

    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    # task 1
    auc_task1 = roc_auc_score(
        y_true[:, 0],
        y_pred[0][:, 1]   # positive class prob
    )

    # task 2
    auc_task2 = roc_auc_score(
        y_true[:, 1],
        y_pred[1][:, 1]
    )

    mean_auc = (auc_task1 + auc_task2) / 2

    return {
        "auc_task1": auc_task1,
        "auc_task2": auc_task2,
        "ROC-AUC": mean_auc
    }