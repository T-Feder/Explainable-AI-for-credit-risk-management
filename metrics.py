from sklearn.metrics import precision_score
import numpy as np

def precision_metric(preds, train_data):
    labels = train_data.get_label()
    preds = np.round(preds)
    return 'precision', precision_score(labels, preds, pos_label=0), True


def precision_metric_classifier(y_true, y_pred):
    return 'precision', precision_score(y_true, np.round(y_pred), pos_label=0), True
