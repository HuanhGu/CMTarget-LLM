from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import numpy as np

def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    if len(np.unique(y_true)) == 1:
        auc = 0.5   # 或者 np.nan
    else:
        auc = roc_auc_score(y_true, y_pred)

    return recall, precision, f1, accuracy, auc