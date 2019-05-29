import numpy as np


class AbstractDataModel(object):
    def __init__(self):
        pass

    def sample_joint_data_points(self, n):
        """
        :return: x, t, y, y_cf, z, sigma
            where - t is assumed to be an element of range(0, num_treatments)
                  - y_cf is assumed to be a list of length num_treatments
        """
        raise NotImplementedError()

    def sample_z(self, x, t):
        raise NotImplementedError()

    def get_prob_z(self, x, t, z_values):
        raise NotImplementedError()

    def get_propensity_score_z(self, z, t):
        raise NotImplementedError()

    def get_propensity_score_x(self, x, t):
        raise NotImplementedError()

    def mu(self, z, t):
        raise NotImplementedError()

    def get_psi(self, x, t):
        raise NotImplementedError()

    def mu_x_t(self, x, t):
        raise NotImplementedError()

    def mu_x(self, x, t):
        raise NotImplementedError()

