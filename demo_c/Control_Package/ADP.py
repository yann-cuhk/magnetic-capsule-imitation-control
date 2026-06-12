import csv
import itertools
import os

import numpy as np


STATE_SIZE = 12
CONTROL_SIZE = 6
DEFAULT_DATA_PATH = "post_processing/adp_samples.csv"
DEFAULT_POLICY_PATH = "Control_Package/adp_capsule_policy.npz"


class ADPDataRecorder:
    def __init__(self, control_info):
        self.path = control_info.get("data_path", DEFAULT_DATA_PATH)
        self.force_std = float(control_info.get("exploration_force_std", 0.0))
        self.moment_std = float(control_info.get("exploration_moment_std", 0.0))
        self.rng = np.random.default_rng(int(control_info.get("exploration_seed", 1)))
        self.rows = 0

        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        if control_info.get("overwrite_data", True) or not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["t"]
                    + [f"x{i}" for i in range(STATE_SIZE)]
                    + [f"u{i}" for i in range(CONTROL_SIZE)]
                )

    def exploration(self):
        force_noise = self.rng.normal(0.0, self.force_std, 3)
        moment_noise = self.rng.normal(0.0, self.moment_std, 3)
        return np.concatenate((force_noise, moment_noise))

    def record(self, t, x, u):
        x = np.asarray(x, dtype=float).reshape(STATE_SIZE)
        u = np.asarray(u, dtype=float).reshape(CONTROL_SIZE)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([float(t)] + x.tolist() + u.tolist())
        self.rows += 1


class ADPPolicy:
    def __init__(self, control_info):
        self.path = control_info.get("policy_path", DEFAULT_POLICY_PATH)
        if not os.path.exists(self.path):
            raise FileNotFoundError(
                f"ADP policy not found: {self.path}. Run solve_adp_capsule.py first."
            )
        data = np.load(self.path, allow_pickle=True)
        self.K = np.asarray(data["K"], dtype=float)
        self.K_behavior = np.asarray(data["K_behavior"], dtype=float) if "K_behavior" in data else self.K
        self.u_min = np.asarray(data["u_min"], dtype=float)
        self.u_max = np.asarray(data["u_max"], dtype=float)
        self.x_min = np.asarray(data["x_min"], dtype=float) if "x_min" in data else None
        self.x_max = np.asarray(data["x_max"], dtype=float) if "x_max" in data else None
        self.blend = float(control_info.get("adp_blend", 0.35))
        self.domain_margin = float(control_info.get("domain_margin", 0.5))
        self.last_mode = "ADP"

        max_force = float(control_info.get("max_force", np.max(np.abs(self.u_max[:3]))))
        max_moment = float(control_info.get("max_moment", np.max(np.abs(self.u_max[3:]))))
        self.u_min[:3] = np.maximum(self.u_min[:3], -max_force)
        self.u_max[:3] = np.minimum(self.u_max[:3], max_force)
        self.u_min[3:] = np.maximum(self.u_min[3:], -max_moment)
        self.u_max[3:] = np.minimum(self.u_max[3:], max_moment)

    def control(self, x):
        x = np.asarray(x, dtype=float).reshape(STATE_SIZE)
        u_adp = self.K @ x
        u_behavior = self.K_behavior @ x
        if self._trust_adp(x, u_adp):
            u = (1.0 - self.blend) * u_behavior + self.blend * u_adp
            self.last_mode = "ADP"
        else:
            u = u_behavior
            self.last_mode = "behavior"
        return np.clip(u, self.u_min, self.u_max)

    def _trust_adp(self, x, u_adp):
        if self.x_min is not None and self.x_max is not None:
            span = np.maximum(self.x_max - self.x_min, 1e-12)
            lower = self.x_min - self.domain_margin * span
            upper = self.x_max + self.domain_margin * span
            if np.any(x < lower) or np.any(x > upper):
                return False

        position_error = x[:3]
        if np.linalg.norm(position_error) > 1e-6 and float(u_adp[:3] @ position_error) < 0.0:
            return False
        return True


def phi_fun(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.concatenate((x, _quadratic_terms(x)))


def psi_fun(x, u):
    xu = np.concatenate(
        (np.asarray(x, dtype=float).reshape(-1), np.asarray(u, dtype=float).reshape(-1))
    )
    return np.concatenate((xu, _quadratic_terms(xu)))


def psi_many(x, u_candidates):
    x = np.asarray(x, dtype=float).reshape(1, -1)
    u_candidates = np.asarray(u_candidates, dtype=float)
    xu = np.hstack((np.repeat(x, len(u_candidates), axis=0), u_candidates))
    idx = np.triu_indices(xu.shape[1])
    return np.hstack((xu, xu[:, idx[0]] * xu[:, idx[1]]))


def load_adp_samples(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size == 0:
        raise ValueError(f"No ADP samples found in {path}")
    data = np.atleast_1d(data)
    t = np.asarray(data["t"], dtype=float)
    x = np.column_stack([np.asarray(data[f"x{i}"], dtype=float) for i in range(STATE_SIZE)])
    u = np.column_stack([np.asarray(data[f"u{i}"], dtype=float) for i in range(CONTROL_SIZE)])
    order = np.argsort(t)
    t, x, u = t[order], x[order], u[order]
    valid = np.isfinite(t) & np.all(np.isfinite(x), axis=1) & np.all(np.isfinite(u), axis=1)
    return t[valid], x[valid], u[valid]


def solve_adp_policy(
    t,
    x,
    u_behavior,
    q_weights=None,
    r_weights=None,
    max_iter=50,
    tol=1e-3,
    nu_grid=3,
    policy_sample_count=800,
    ridge=1e-8,
):
    t = np.asarray(t, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float)
    u_behavior = np.asarray(u_behavior, dtype=float)
    if len(t) < 3:
        raise ValueError("ADP needs at least three time samples")
    if x.shape[1] != STATE_SIZE or u_behavior.shape[1] != CONTROL_SIZE:
        raise ValueError("Unexpected ADP sample dimensions")

    q_weights = np.asarray(q_weights if q_weights is not None else _default_q(), dtype=float)
    r_weights = np.asarray(r_weights if r_weights is not None else _default_r(), dtype=float)
    Q = np.diag(q_weights)
    R = np.diag(r_weights)

    K_behavior = np.linalg.lstsq(x, u_behavior, rcond=None)[0].T
    K = K_behavior.copy()
    u_min = np.min(u_behavior, axis=0)
    u_max = np.max(u_behavior, axis=0)
    u_min, u_max = _expand_bounds(u_min, u_max)
    candidates = _control_candidates(u_min, u_max, nu_grid)
    sample_idx = _policy_sample_indices(len(x), policy_sample_count)
    converged = False
    w = None
    c = None

    for iteration in range(1, max_iter + 1):
        A_rows = []
        b_rows = []
        for i in range(len(t) - 1):
            dt = t[i + 1] - t[i]
            if dt <= 0:
                continue
            x0 = x[i]
            x1 = x[i + 1]
            ub0 = u_behavior[i]
            ub1 = u_behavior[i + 1]
            uc0 = K @ x0
            uc1 = K @ x1

            delta_phi = phi_fun(x1) - phi_fun(x0)
            psi_delta = (
                psi_fun(x0, uc0)
                - psi_fun(x0, ub0)
                + psi_fun(x1, uc1)
                - psi_fun(x1, ub1)
            ) * dt / 2.0
            cost = -(
                x0 @ Q @ x0
                + uc0 @ R @ uc0
                + x1 @ Q @ x1
                + uc1 @ R @ uc1
            ) * dt / 2.0
            A_rows.append(np.concatenate((delta_phi, psi_delta)))
            b_rows.append(cost)

        A = np.asarray(A_rows, dtype=float)
        b = np.asarray(b_rows, dtype=float)
        pp = _least_squares(A, b, ridge)
        n_phi = len(phi_fun(np.zeros(STATE_SIZE)))
        w = pp[:n_phi]
        c = pp[n_phi:]

        x_sample = x[sample_idx]
        optimal_u = np.zeros((len(x_sample), CONTROL_SIZE))
        for row, xi in enumerate(x_sample):
            costs = psi_many(xi, candidates) @ c
            costs += xi @ Q @ xi
            costs += np.einsum("ij,jk,ik->i", candidates, R, candidates)
            optimal_u[row] = candidates[int(np.argmin(costs))]

        K_new = np.linalg.lstsq(x_sample, optimal_u, rcond=None)[0].T
        if np.linalg.norm(K_new - K) < tol:
            K = K_new
            converged = True
            break
        K = K_new

    return {
        "K": K,
        "K_behavior": K_behavior,
        "w": w,
        "c": c,
        "q_weights": q_weights,
        "r_weights": r_weights,
        "u_min": u_min,
        "u_max": u_max,
        "x_min": np.min(x, axis=0),
        "x_max": np.max(x, axis=0),
        "iterations": iteration,
        "converged": converged,
        "sample_count": len(t),
    }


def save_policy(path, result):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **result)


def _quadratic_terms(values):
    idx = np.triu_indices(len(values))
    return values[idx[0]] * values[idx[1]]


def _least_squares(A, b, ridge):
    if ridge and ridge > 0:
        lhs = A.T @ A + ridge * np.eye(A.shape[1])
        rhs = A.T @ b
        return np.linalg.solve(lhs, rhs)
    return np.linalg.lstsq(A, b, rcond=None)[0]


def _control_candidates(u_min, u_max, nu_grid):
    axes = [np.linspace(u_min[i], u_max[i], int(nu_grid)) for i in range(CONTROL_SIZE)]
    return np.asarray(list(itertools.product(*axes)), dtype=float)


def _policy_sample_indices(length, count):
    if length <= count:
        return np.arange(length)
    return np.linspace(0, length - 1, int(count)).astype(int)


def _expand_bounds(u_min, u_max):
    span = np.maximum(u_max - u_min, 1e-12)
    return u_min - 0.1 * span, u_max + 0.1 * span


def _default_q():
    return [50.0, 50.0, 50.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1]


def _default_r():
    return [1.0e4, 1.0e4, 1.0e4, 1.0e8, 1.0e8, 1.0e8]
