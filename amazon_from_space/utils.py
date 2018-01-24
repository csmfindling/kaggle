from theano import config
import numpy as np


def f2_score(prediction, target, beta=2):
    beta = np.float32(beta)
    true_positives = (prediction * target).sum().astype(config.floatX)
    all_pred_positives = prediction.sum().astype(config.floatX)
    actual_positives = target.sum().astype(config.floatX)
    p = true_positives / all_pred_positives
    r = true_positives / actual_positives
    f2 = p * r / (beta**2 * p + r)
    f2 *= (np.float32(1) + beta**2)
    return f2
