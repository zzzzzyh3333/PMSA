"""ACC, relative forgetting (RF), and performance-aware RF (PRF).

Values are fractions. The peak excludes the final stage and improvements are
clamped to zero forgetting, matching the original positive-peak definitions.
RF/PRF are undefined (None) for one stage or an old domain with zero peak.
"""
import math


def _validate(matrix):
    if not matrix or any(len(row) != i + 1 for i, row in enumerate(matrix)):
        raise ValueError('Expected a nonempty triangular accuracy matrix')
    if any(not math.isfinite(v) or not 0 <= v <= 1 for row in matrix for v in row):
        raise ValueError('Accuracy values must be finite fractions in [0, 1]')


def average_accuracy(acc_matrix):
    _validate(acc_matrix)
    return sum(acc_matrix[-1]) / len(acc_matrix)


def _forgetting(acc_matrix, performance_aware):
    _validate(acc_matrix)
    n = len(acc_matrix)
    if n < 2:
        return None
    values = []
    for j in range(n - 1):
        peak = max(acc_matrix[i][j] for i in range(j, n - 1))
        if peak == 0:
            return None
        final = acc_matrix[-1][j]
        value = max((peak - final) / peak, 0.0)
        values.append((1 - final) * value if performance_aware else value)
    return sum(values) / len(values)


def forgetting(acc_matrix):
    return _forgetting(acc_matrix, False)


def performance_aware_forgetting(acc_matrix):
    return _forgetting(acc_matrix, True)


def metrics(acc_matrix):
    return {'ACC': average_accuracy(acc_matrix), 'RF': forgetting(acc_matrix),
            'PRF': performance_aware_forgetting(acc_matrix)}
