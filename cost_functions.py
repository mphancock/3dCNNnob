import numpy as np

def dice_coeff(r1, r2):
    true_f = r1.flatten()
    pred_f = r2.flatten()
    intersect = np.sum(true_f*pred_f)
    return (2 * intersect) / (np.sum(true_f) + np.sum(pred_f))