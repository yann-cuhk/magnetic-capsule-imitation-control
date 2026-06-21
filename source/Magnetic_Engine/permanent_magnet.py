import numpy as np


def _column3(value):
    return np.asmatrix(np.asarray(value, dtype=float).reshape(3, 1))


def _scalar(value):
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def force_moment(p, ma, mc):
    k = 4 * np.pi * 1e-7
    ma = _column3(ma)
    mc = _column3(mc)
    p = _column3(p)
    p_hat = p / np.linalg.norm(p)
    p_norm = float(np.linalg.norm(p))
    field = k / (4 * np.pi * pow(p_norm, 5)) * (3 * (p * p.T) - pow(p_norm, 2) * np.eye(3)) * ma
    field_jacobian = 3 * k / (4 * np.pi * pow(p_norm, 4)) * (
        ma * p_hat.T
        + p_hat * ma.T
        + _scalar(p_hat.T * ma) * (np.eye(3) - 5 * (p_hat * p_hat.T))
    )
    S_mc = np.asmatrix(
        [
            [0, -mc[2, 0], mc[1, 0]],
            [mc[2, 0], 0, -mc[0, 0]],
            [-mc[1, 0], mc[0, 0], 0],
        ]
    )
    force = field_jacobian * mc
    moment = S_mc * field
    return force, moment


def force_moment_jacob(p, ma, mc):
    X = np.asmatrix(np.concatenate((_column3(p), _column3(ma), _column3(mc)), axis=0))
    h = 0.00001
    jacobian = np.asmatrix(np.zeros([6, 9]))
    for idx in range(X.size):
        tmp_val = X[idx, 0]
        X[idx, 0] = tmp_val + h
        force1, moment1 = force_moment(X[0:3, 0], X[3:6, 0], X[6:9, 0])
        X[idx, 0] = tmp_val - h
        force2, moment2 = force_moment(X[0:3, 0], X[3:6, 0], X[6:9, 0])
        jacobian[:3, idx] = (force1 - force2) / (2 * h)
        jacobian[3:6, idx] = (moment1 - moment2) / (2 * h)
        X[idx, 0] = tmp_val
    return jacobian
