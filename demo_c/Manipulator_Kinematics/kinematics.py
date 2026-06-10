import numpy as np
from math import sin, cos
from scipy.optimize import fsolve, least_squares
from Pose_Transform.pose_transform import *


def wrap_to_pi(theta):
    return (np.asarray(theta, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def T_param(theta, a, d, alpha):
    """
    获得相邻连杆之间的齐次变换矩阵函数(具体参数值参考机械臂的DH参数表)
    :param theta: 关节转角 float
    :param a: 连杆长度 float
    :param d: 连杆偏移 float
    :param alpha: 连杆扭转角 float
    :return: 齐次变换矩阵 np.mat 4*4
    """
    T = np.mat([[cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
                [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
                [0, sin(alpha), cos(alpha), d],
                [0, 0, 0, 1]])
    return T

class Manipulator2:
    def __init__(self, base_pose, a, d, alpha, T_bias):
        """
        :param base_matrix: 基坐标系关于世界坐标系的齐次变换矩阵 np.array/np.mat 4*4
        :param a: 连杆长度 array n*1
        :param d: 连杆偏移 array n*1
        :param alpha: 连杆扭转角 array n*1
        :param bias_length: 偏差（永磁铁长度的一半） float
        """
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

    # def set_theta(self, theta):
    #     """
    #     输入theta值，设置机械臂的关节角度为theta
    #     :param theta: 关节角度 np.array n*1
    #     :return: 无返回值
    #     """
    #     self.theta = theta

    def fkine(self, theta):
        """
        整个机械臂的前向运动学,关节角度theta映射到末端永磁铁的位置和磁矩（默认永磁铁磁矩和末端坐标系的z轴平行）
        :return: 齐次变化矩阵 np.mat 4*4
        """
        T_matrix = self.base_matrix
        for i in range(theta.size):
            T_matrix = T_matrix * T_param(theta[i, 0], self.a[i, 0], self.d[i, 0], self.alpha[i, 0])

        T_matrix = T_matrix * self.T_bias

        p = T_matrix[0:3, 3]
        m_hat = T_matrix[0:3, 0:3] * np.mat([[1], [0], [0]])

        return p, m_hat

    def fkine_all_link(self, theta):
        """
        求前向运动学过程中各个连杆的位置和姿态四元数
        :return: 二维list n*7
        """
        link_list = list()
        T_matrix = self.base_matrix
        # print(T_matrix)
        # print(type(T_matrix))
        # print(T_matrix_to_pose(T_matrix))
        link_list.append(T_matrix_to_pose(T_matrix))

        for i in range(theta.size):
            T_matrix = T_matrix * T_param(theta[i, 0], self.a[i, 0], self.d[i, 0], self.alpha[i, 0])
            link_list.append(T_matrix_to_pose(T_matrix))

        T_matrix = T_matrix * self.T_bias
        link_list.append(T_matrix_to_pose(T_matrix))

        return link_list

    def fkine_jacob(self, theta):
        """
        求机械臂关节到末端位置和磁矩的雅可比矩阵
        :param theta: 关节角度 np.array n*1 (theta里面数据需为浮点数，不然可能会出问题)
        :return: 雅可比矩阵 np.array 6*n
        """
        X = np.mat(theta)
        h = 1e-4
        grad = np.mat(np.zeros([6, 6]))
        # print(X)
        for idx in range(X.size):
            tmp_val = float(X[idx, 0])
            # print(tmp_val + h)
            X[idx, 0] = tmp_val + h
            # print(float(X[idx, 0]))
            # print(f"第{idx+1}个的X+h为{X[idx, 0]}")
            p1, m_hat1 = self.fkine(X)
            X[idx, 0] = tmp_val - h
            # print(f"第{idx+1}个的X-h为{X[idx, 0]}")
            p2, m_hat2 = self.fkine(X)

            grad[:3, idx] = (p1 - p2) / (2 * h)
            grad[3:6, idx] = (m_hat1 - m_hat2) / (2 * h)
            X[idx, 0] = tmp_val
            # print(f"第{idx+1}个的X为{X[idx, 0]}")
        return grad

    def __back_function(self, theta):
        """
        逆解函数(私有函数)
        :param theta: 关节角度 np.array n*1
        :return: 待解列表 == 0
        """
        theta = np.array([[theta[0]], [theta[1]], [theta[2]], [theta[3]], [theta[4]], [theta[5]]])
        p, m_hat = self.fkine(theta)
        p_new = p - self.p
        m_hat_new = m_hat - self.m_hat

        return [p_new[0, 0], p_new[1, 0], p_new[2, 0], m_hat_new[0, 0], m_hat_new[1, 0], m_hat_new[2, 0]]

    def back_numerical(self, p, m_hat, X0):
        """
        求机械臂逆运动学解的函数
        :param p: 末端永磁铁的位置(中心位置) np.array 3*1
        :param m_hat: 末端永磁铁的磁矩方向(单位向量) np.array 3*1
        :param X0: 迭代初始值 一级列表list length==6
        :return: 逆解值 一级列表list length==6
        """
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

if __name__ == "__main__":
    T0 = np.array([[1, 0, 0, -0.4], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    a = np.array([[0], [-0.42500], [-0.39225], [0], [0], [0]])
    d = np.array([[0.089159], [0], [0], [0.10915], [0.09465], [0.08230]])
    alpha = np.array([[np.pi / 2], [0], [0], [np.pi / 2], [-np.pi / 2], [0]])
    bias = 0.03175 / 2
    aa = Manipulator(T0, a, d, alpha, bias)
    theta = np.array([[0.5], [1], [1.5], [2.0], [3.1], [0.0]])
    aa.set_theta(theta)
    p = np.array([[0.12], [0], [0.32]])
    m_hat = np.array([[0], [0], [-1]])
    X0 = [1, -1, -1, 1, 1, 1]
    # print(aa.back_numerical(p, m_hat, X0))

    print(aa.fkine(theta))
    print(aa.fkine_jacob(theta))

    # tmp = aa.Tmatrix_to_pose(T0)
    # print(tmp)
    # print((type(tmp)))
    # print(tmp.shape)

    # print(aa.fkine_all_link())
    # print(len(aa.fkine_all_link()))
    # print(type(aa.fkine_all_link(theta)))
    # print(len(aa.fkine_all_link(theta)))
    # print(type(aa.fkine_all_link(theta)[0]))
