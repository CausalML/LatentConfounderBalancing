import numpy as np
from scipy.spatial.distance import cdist
import quadprog

from weights_learning.abstract_weights_learning import AbstractWeightsLearning


class BalancedWeightsLearningContinuousQuadprogX(AbstractWeightsLearning):
    def __init__(self, num_data, num_treatment, data_model, kernel):
        AbstractWeightsLearning.__init__(self, num_data, num_treatment,
                                         data_model)
        self.kernel = kernel

    def train(self, x, t, y, verbose=False, normalized_weights=True,
              max_mu_norm=None, sigma=None, num_sample=1000):
        self.check_data(x, t, y)
        nn = self.num_data
        nt = self.num_treatment

        # calculate q array

        # print("making q matrix")
        # q = np.zeros((nn, nn))
        # for i in range(nn):
        #     for j in range(nn):
        #         q[i, j] += self.kernel(x[i], x[j])
        if self.kernel == "rbf":
            dists = cdist(x, x, "euclidean")
            q = np.exp(-0.5 * dists ** 2)
        else:
            q = np.zeros(shape=(nn, nn))
            for i in range(nn):
                for j in range(nn):
                    q[i, j] = self.kernel(x[i], x[j])

        q = (q + q.T) / 2

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

