import numpy as np


def apply_correction_pipeline(x, powers_, mean_, var_, coef_, intercept_):
    """
    Pure Python/NumPy equivalent of apply_correction_pipeline_cpp
    from Pupil Labs' refraction correction pipeline.
    """

    # Ensure all are numpy arrays (float64)
    x = np.ascontiguousarray(x, dtype=float)
    powers_ = np.ascontiguousarray(powers_, dtype=float)
    mean_ = np.ascontiguousarray(mean_, dtype=float)
    var_ = np.ascontiguousarray(var_, dtype=float)
    coef_ = np.ascontiguousarray(coef_, dtype=float)
    intercept_ = np.ascontiguousarray(intercept_, dtype=float)

    x = x.T
    powers_ = powers_.T
    mean_ = mean_.T
    var_ = var_.T
    coef_ = coef_.T
    intercept_ = intercept_.T

    # features = zeros(powers_.rows(), x.rows())
    features = np.zeros((powers_.shape[0], x.shape[0]), dtype=float)

    # Compute polynomial features
    for k in range(x.shape[0]):          # for each sample (rows of x)
        for i in range(powers_.shape[0]):  # for each power row
            val = 1.0
            for j in range(powers_.shape[1]):
                exponent = int(powers_[i, j])
                if exponent > 0:
                    val *= x[k, j] ** exponent
            val -= mean_[0, i]
            val /= np.sqrt(var_[0, i])
            features[i, k] = val

    result = coef_ @ features
    result += intercept_[:, [0]]  # add intercept columnwise
    result = result.T

    return result
