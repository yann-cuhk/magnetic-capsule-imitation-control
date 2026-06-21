import numpy as np

class Sofa_Interface:

    def set_force_moment(self, child, force, moment):
        child.forces[0][:] = [force[0, 0], force[1, 0], force[2, 0], moment[0, 0], moment[1, 0], moment[2, 0]]

    def get_velocity(self, child):
        return np.array([[child[0]], [child[1]], [child[2]], [0], [0], [0]])
