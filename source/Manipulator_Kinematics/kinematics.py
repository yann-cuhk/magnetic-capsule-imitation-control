import numpy as np
from math import sin, cos
from scipy.optimize import fsolve, least_squares
from Pose_Transform.pose_transform import pose_to_T_matrix, T_matrix_to_pose

def wrap_to_pi(theta):
    return (np.asarray(theta, dtype=float) + np.pi) % (2 * np.pi) - np.pi

def T_param(theta, a, d, alpha):
    T = np.mat([[cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
                [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
                [0, sin(alpha), cos(alpha), d],
                [0, 0, 0, 1]])
    return T

class Manipulator2:

    def __init__(self, base_pose, a, d, alpha, T_bias):
        self.base_matrix = np.mat(pose_to_T_matrix(base_pose))
        self.a = np.mat(a)
        self.d = np.mat(d)
        self.alpha = np.mat(alpha)
        self.T_bias = T_bias
        self.p = None
        self.m_hat = None
        self.joint_lower = -np.pi * np.ones(6)
        self.joint_upper = np.pi * np.ones(6)
        self.joint_reference = np.deg2rad([170.0, 0.0, 0.0, 30.0, 120.0, 20.0])

    def fkine(self, theta):
        T_matrix = self.base_matrix
        for i in range(theta.size):
            T_matrix = T_matrix * T_param(theta[i, 0], self.a[i, 0], self.d[i, 0], self.alpha[i, 0])
        T_matrix = T_matrix * self.T_bias
        p = T_matrix[0:3, 3]
        m_hat = T_matrix[0:3, 0:3] * np.mat([[1], [0], [0]])
        return p, m_hat

    def fkine_all_link(self, theta):
        link_list = list()
        T_matrix = self.base_matrix
        link_list.append(T_matrix_to_pose(T_matrix))
        for i in range(theta.size):
            T_matrix = T_matrix * T_param(theta[i, 0], self.a[i, 0], self.d[i, 0], self.alpha[i, 0])
            link_list.append(T_matrix_to_pose(T_matrix))
        T_matrix = T_matrix * self.T_bias
        link_list.append(T_matrix_to_pose(T_matrix))
        return link_list

    def fkine_jacob(self, theta):
        X = np.mat(theta)
        h = 1e-4
        grad = np.mat(np.zeros([6, 6]))
        for idx in range(X.size):
            tmp_val = float(X[idx, 0])
            X[idx, 0] = tmp_val + h
            p1, m_hat1 = self.fkine(X)
            X[idx, 0] = tmp_val - h
            p2, m_hat2 = self.fkine(X)
            grad[:3, idx] = (p1 - p2) / (2 * h)
            grad[3:6, idx] = (m_hat1 - m_hat2) / (2 * h)
            X[idx, 0] = tmp_val
        return grad

    def __back_function(self, theta):
        theta = np.array([[theta[0]], [theta[1]], [theta[2]], [theta[3]], [theta[4]], [theta[5]]])
        p, m_hat = self.fkine(theta)
        p_new = p - self.p
        m_hat_new = m_hat - self.m_hat
        return [p_new[0, 0], p_new[1, 0], p_new[2, 0], m_hat_new[0, 0], m_hat_new[1, 0], m_hat_new[2, 0]]

    def back_numerical(self, p, m_hat, X0):
        self.p = p
        self.m_hat = m_hat
        p_target = np.asarray(p, dtype=float).reshape(3)
        m_target = np.asarray(m_hat, dtype=float).reshape(3)
        x0 = np.asarray(X0, dtype=float).reshape(6)
        x0 = np.clip(wrap_to_pi(x0), self.joint_lower + 1e-6, self.joint_upper - 1e-6)
        seeds = [
            x0,
            self.joint_reference,
            np.zeros(6),
            self.joint_reference,
            np.deg2rad([170.0, 0.0, 0.0, 30.0, 120.0, 20.0]),
            np.deg2rad([170.0, 5.0, 5.0, 30.0, 120.0, 20.0]),
            np.deg2rad([165.0, 20.0, 5.0, -100.0, 100.0, -30.0]),
            np.array([0.0, 3.0, 0.2, 2.5, -0.8, 0.5]),
            np.array([0.0, -3.0, -0.2, 2.6, -0.8, 0.5]),
        ]
        def residual(theta):
            theta_mat = np.asarray(theta, dtype=float).reshape(6, 1)
            p_now, m_now = self.fkine(theta_mat)
            p_err = np.asarray(p_now, dtype=float).reshape(3) - p_target
            m_err = np.asarray(m_now, dtype=float).reshape(3) - m_target
            return np.concatenate((200.0 * p_err, 10.0 * m_err))
        best = None
        for seed in seeds:
            seed = np.clip(wrap_to_pi(seed), self.joint_lower + 1e-6, self.joint_upper - 1e-6)
            solve = least_squares(
                residual,
                seed,
                bounds=(self.joint_lower, self.joint_upper),
                max_nfev=300,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
            )
            theta = wrap_to_pi(solve.x)
            p_now, m_now = self.fkine(theta.reshape(6, 1))
            p_err = np.linalg.norm(np.asarray(p_now, dtype=float).reshape(3) - p_target)
            m_err = np.linalg.norm(np.asarray(m_now, dtype=float).reshape(3) - m_target)
            posture_score = np.linalg.norm(wrap_to_pi(theta - self.joint_reference))
            score = posture_score + 1000.0 * max(p_err - 1e-5, 0.0) + 10.0 * max(m_err - 1e-4, 0.0)
            if best is None or score < best[0]:
                best = (score, theta, p_err, m_err)
        if best is not None and best[2] < 1e-4 and best[3] < 1e-3:
            result_theta = best[1].reshape(6, 1)
        else:
            result = list(fsolve(self.__back_function, X0))
            result_theta = wrap_to_pi(result).reshape(6, 1)
        return result_theta
