import numpy as np

def augment_cdf(x, x_a, y):
    i_shift = 0
    y_a = y.copy()
    for i in range(len(x) - 1):
        i1 = len(np.where(x_a > x[i])[0]) + len(np.where(x_a < x[i + 1])[0]) - len(x_a)
        if i1 > 0:
            ones_matrix = np.ones((i1, y_a.shape[1]))
            y_a = np.concatenate((y_a[:i + i_shift + 1, :], y_a[i + i_shift, :] * ones_matrix, y_a[i + i_shift + 1:, :]), axis=0)
            i_shift += i1
    ia = np.where(x_a < x[0])[0]
    y_a = np.concatenate((np.zeros((len(ia), y_a.shape[1])), y_a), axis=0)
    ie = np.where(x_a > x[-1])[0]
    y_a = np.concatenate((y_a, np.ones((len(ie), y_a.shape[1]))), axis=0)
    return y_a
