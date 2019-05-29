import numpy as np
from scipy.spatial.distance import cdist
import quadprog

from utils.hide_output import HideOutput

from weights_learning.abstract_weights_learning import AbstractWeightsLearning


class BalancedWeightsLearningContinuousQuadprog(AbstractWeightsLearning):
    def __init__(self, num_data, num_treatment, data_model, kernel):
        AbstractWeightsLearning.__init__(self, num_data, num_treatment,
                                         data_model)
        self.kernel = kernel

    def train(self, x, t, y, verbose=False, normalized_weights=True,
              max_mu_norm=None, sigma=None, num_sample=50, z_sample=None):
        self.check_data(x, t, y)
        nn = self.num_data
        nt = self.num_treatment

        # calculate q array

        # z_sample is of shape (nn, num_sample*2, dim_z)
        if z_sample is None:
            with HideOutput():
                z_sample = self.data_model.sample_z(
                    x, t, num_sample=num_sample, thin=5)
        # z_sample = np.random.randn(nn, num_sample, 2)

        # print("making q matrix")
        # q = np.zeros((nn, nn))
        # for b in range(num_sample):
        #     for i in range(nn):
        #         for j in range(nn):
        #             q[i, j] += self.kernel(z_sample_1[i, b], z_sample_2[j, b])
        # q = (q + q.T) / (2 * num_sample)
        # z_sample_1 = z_sample_1.reshape(-1, z_sample_1.shape[2])
        # z_sample_2 = z_sample_2.reshape(-1, z_sample_2.shape[2])
        # dists = cdist(z_sample_1, z_sample_2, "euclidean").reshape(nn, num_sample, nn, num_sample).mean(1).mean(2)
        # q = np.exp(-0.5 * dists ** 2)
        q = np.zeros((nn, nn))
        z_sample_flat = z_sample.reshape(-1, z_sample.shape[2])
        for b in range(num_sample):
            if self.kernel == "rbf":
                dists = cdist(z_sample_flat, z_sample[:, b, :], "euclidean")
                q += np.exp(-0.5 * (dists ** 2)).reshape(nn, -1, nn).sum(1)
            else:
                for i in range(nn):
                    for j in range(nn):
                        for c in range(num_sample):
                            q[i, j] += self.kernel(z_sample[i, b],
                                                   z_sample[j, c])
        # q = np.zeros((nn, nn))
        # for b1 in range(num_sample):
        #     for b2 in range(num_sample):
        #         dists = cdist(z_sample_1[:, b1, :], z_sample_2[:, b2, :],
        #                       "euclidean")
        #     q += np.exp(-0.5 * dists ** 2)
        # q = (q + q.T) / (2 * num_sample ** 2)
        # q = (q + q.T) / 2
        q = (q + q.T) / (2 * (num_sample ** 2))

        delta = t.reshape(1, -1) == t.reshape(-1, 1)
        if sigma is None:
            sigma = np.ones(nn)
        pi_t_x = np.zeros((nn, nn))
        for i in range(nn):
            for j in range(nn):
                pi_t_x[i, j] = self.policy(x[i], t[j])

        G = 2 * (delta * q + np.diag(sigma))
        a = 2 * (q * pi_t_x).sum(0)

        pi_vec_list = []
        for t in range(nt):
            pi_vec = np.array([self.policy(x_val, t) for x_val in x])
            pi_vec_list.append(pi_vec)
        pi_matrix = (np.array(pi_vec_list).reshape(nt, 1, -1)
                     * np.array(pi_vec_list).reshape(nt, -1, 1)).sum(0)
        bias = (pi_matrix * q).sum()

        # solve quadratic program:
        #   min     0.5 w^T G w - a^T w
        #   st      C.T w >= b
        if normalized_weights is False:
            w, obj, wu, iter, _, _ = quadprog.solve_qp(G=G, a=a)
        else:
            C = np.concatenate([np.ones(nn).reshape(1, -1),
                                np.identity(nn)]).T
            b = np.zeros(nn + 1)
            b[0] = nn
            w, obj, wu, iter, _, _ = quadprog.solve_qp(G=G, a=a, C=C, b=b,
                                                       meq=1)

        # print("w:", w)
        cmse = (obj + bias) / (nn ** 2)
        # print("CMSE:", cmse)
        self.meta_data = {
            "min_obj": cmse,
        }
        # print("w unconstrained:", wu)
        # print("num iter:", iter)
        return w

