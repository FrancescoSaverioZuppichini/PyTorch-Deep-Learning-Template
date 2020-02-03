import torch
from torchbearer import metrics

# @metrics.default_for_key('acc')
# @metrics.mean
# @metrics.lambda_metric('acc', on_epoch=False)
# def categorical_accuracy(y_pred, y_true):
#    _, y_pred = torch.max(y_pred, 1)
#    return (y_pred == y_true).float()