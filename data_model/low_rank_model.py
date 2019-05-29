import pystan
from data_model.abstract_data_model import AbstractDataModel

import numpy as np
from scipy import stats
from numpy.linalg import inv
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim


class LowRankModel(AbstractDataModel):
    def __init__(self, dim_z, dim_x, num_t, v, v_0, std_x,
                 p, p_0, gamma, gamma_0, std_y, y_activation=None):
        AbstractDataModel.__init__(self)
        self.dim_z = dim_z
        self.dim_x = dim_x
        self.num_t = num_t
        self.v = v
        self.v_0 = v_0
        self.std_x = std_x
        self.p = p
        self.p_0 = p_0
        self.gamma = gamma
        self.gamma_0 = gamma_0
        self.std_y = std_y
        assert v.shape == (dim_z, dim_x)
        assert v_0.shape == std_x.shape == (dim_x,)
        assert p.shape == (dim_z, num_t)
        assert p_0.shape == (num_t,)
        assert gamma.shape == (dim_z, num_t)
        assert gamma_0.shape == std_y.shape == (num_t,)

        if y_activation is None:
            self.y_activation = lambda x_: x_
        else:
            self.y_activation = y_activation

        self.q_network = QNetwork(dim_x, num_t)
        self._train_q_network(verbose=False)

        id_x = np.identity(dim_x)
        id_z = np.identity(dim_z)
        self.x_cov = inv(id_x - v.T @ inv(v @ v.T + id_z) @ v)

        self.stan_model = None

    def sample_joint_data_points(self, n):
        # sample z
        z = np.random.normal(size=(n, self.dim_z))

        # sample x
        mean_x = z @ self.v + self.v_0.reshape(1, self.dim_x)
        x = np.random.normal(mean_x, self.std_x.reshape(1, -1))

        # sample t
        t_values = list(range(self.num_t))
        t_scores = np.exp(z @ self.p + self.p_0)
        t_probs = t_scores / t_scores.sum(1, keepdims=True)
        t = np.array([np.random.choice(t_values, p=p) for p in t_probs])

        # sample y
        y_cf_mu = self.y_activation(z @ self.gamma + self.gamma_0.reshape(1, self.num_t))
        y_cf = np.random.normal(y_cf_mu, self.std_y.reshape(1, -1))
        y = np.array([y_cf[i, t_] for i, t_ in enumerate(t)])
        sigma = np.array([self.std_y[t_] for t_ in t])

        return x, t, y, y_cf, z, sigma

    def mu(self, z, t):
        mu_all = self.y_activation(z @ self.gamma + self.gamma_0)
        return mu_all[t]

    def mu_x_t(self, x, t):
        # return expectation of mu_t(Z_i) given X_i and T_i, where t = T_i
        raise NotImplementedError()

    def mu_x(self, x, t):
        # return expectation of mu_t(Z_i) given X_i, where t = T_i
        raise NotImplementedError()

    def sample_z_unconditional(self, num_z):
        sample = np.random.normal(size=(num_z, self.dim_z))
        z_mean = np.zeros(self.dim_z)
        sample_probs = stats.multivariate_normal.pdf(sample, mean=z_mean)
        return sample, sample_probs

    def sample_z(self, x, t, warmup=500, chains=2, num_sample=2000, thin=2):
        total_sample = warmup * chains + num_sample * thin
        num_iter = int(total_sample / chains)
        fit = self._sample_z_stan(x, t, num_iter=num_iter,
                                  chains=chains, warmup=warmup, thin=thin)
        return fit.extract("z")["z"].transpose(1, 0, 2)

    def sample_z_mode(self, x, t, warmup=500, chains=2, num_sample=2000):
        total_sample = warmup * chains + num_sample
        num_iter = int(total_sample / chains)
        fit = self._sample_z_stan(x, t, num_iter=num_iter,
                                  chains=chains, warmup=warmup)
        params = fit.extract(permuted=True)
        a_i = params["lp__"].argmax()
        return params["z"][a_i]

    def _sample_z_stan(self, x, t, num_iter=1000, chains=2, warmup=500, thin=1):
        if self.stan_model is None:
            self.stan_model = pystan.StanModel("continuous_model.stan")
        standata = {
            "N": x.shape[0], "Z": self.dim_z, "X": self.dim_x, "T": self.num_t,
            "R": self.v.T, "R_0": self.v_0, "std_x": self.std_x,
            "P": self.p.T, "P_0": self.p_0,
            "x": x, "t": t + 1}
        fit = self.stan_model.sampling(
            data=standata, iter=num_iter, warmup=warmup, thin=thin,
            chains=chains, verbose=False)
        return fit

    def get_prob_z(self, x, t, z_values):
        # return array of probability of each provided Z value given X_i, T_i
        #   array of shape (nn, nz)
        pr_x_t_given_z = self.get_prob_x_t_given_z(x, t, z_values)
        z_mean = np.zeros(self.dim_z)
        pr_z = np.array([stats.multivariate_normal.pdf(z, mean=z_mean)
                         for z in z_values]).reshape(1, -1)
        pr_x_t = self.estimate_pr_x_t(x, t).reshape(-1, 1)
        return pr_x_t_given_z * pr_z / pr_x_t

    def get_prob_x_t_given_z(self, x, t, z_values):
        mean_x_sample = z_values @ self.v + self.v_0.reshape(1, self.dim_x)
        x_probs_all = np.array([stats.norm.pdf(x, loc=mean_x, scale=self.std_x)
                                for mean_x in mean_x_sample]).prod(2)

        t_scores_sample = np.exp(z_values @ self.p + self.p_0)
        t_probs_sample = t_scores_sample / t_scores_sample.sum(1, keepdims=True)
        t_probs_all = np.array([[t_probs[t_] for t_ in t]
                                for t_probs in t_probs_sample])

        return (x_probs_all * t_probs_all).transpose(1, 0)

    def get_propensity_score_z(self, z, t):
        t_scores_all = np.exp(z @ self.p + self.p_0)
        t_probs_all = t_scores_all / t_scores_all.sum(1, keepdims=True)
        return np.array([t_probs[t_] for t_probs, t_ in zip(t_probs_all, t)])

    def get_propensity_score_x(self, x, t):
        t_probs_all = self.estimate_prob_t_given_x(x)
        return np.array([t_probs[t_] for t_probs, t_ in zip(t_probs_all, t)])

    def estimate_pr_x_t(self, x, t):
        pr_t_given_x = self.get_propensity_score_x(x, t)
        pr_x = stats.multivariate_normal.pdf(x, mean=self.v_0, cov=self.x_cov)
        return pr_t_given_x * pr_x

    def estimate_prob_t_given_x(self, x):
        log_probs = self.q_network(torch.from_numpy(x).float())
        return torch.exp(log_probs).detach().numpy()

    def _train_q_network(self, batch_size=32, num_iter=2000, verbose=True):
        self.q_network = QNetwork(self.dim_x, self.num_t)
        optimizer = optim.Adam(self.q_network.parameters())

        loss_history = []
        for i in range(num_iter):
            if verbose and i % 100 == 0:
                if loss_history:
                    mean_loss = float(np.mean(loss_history))
                else:
                    mean_loss = float("nan")
                print("iter=%d, mean_loss=%r" % (i, mean_loss))
                loss_history = []
            optimizer.zero_grad()

            # sample batch data
            z = np.random.normal(size=(batch_size, self.dim_z))
            mean_x = z @ self.v + self.v_0.reshape(1, self.dim_x)
            x = np.random.normal(mean_x, self.std_x.reshape(1, -1))
            t_scores = np.exp(z @ self.p + self.p_0)
            target_probs = t_scores / t_scores.sum(1, keepdims=True)
            target_probs = torch.from_numpy(target_probs).float()

            # obtain loss
            output_log_probs = self.q_network(torch.from_numpy(x).float())
            loss = -torch.sum(target_probs * output_log_probs)
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss))


class QNetwork(torch.nn.Module):

    def __init__(self, x_dim, num_t):
        torch.nn.Module.__init__(self)
        self.x_dim = x_dim
        self.num_t = num_t
        self.linear_1 = torch.nn.Linear(x_dim, 200)
        self.linear_2 = torch.nn.Linear(200, 200)
        self.linear_3 = torch.nn.Linear(200, num_t)

    def forward(self, x):
        h = torch.sigmoid(self.linear_1(x))
        h = torch.sigmoid(self.linear_2(h))
        return F.log_softmax(self.linear_3(h), dim=1)




def debug():
    dim_z = 2
    dim_x = 4
    num_t = 2
    v = np.random.randn(2, 4)
    v_0 = np.random.randn(4)
    std_x = np.ones(4)
    p = np.random.randn(2, 2)
    p_0 = np.random.randn(2)
    gamma = np.random.randn(2, 2)
    gamma_0 = np.random.randn(2)
    std_y = np.ones(2)
    d = LowRankModel(dim_z=dim_z, dim_x=dim_x, num_t=num_t, v=v, v_0=v_0,
                     std_x=std_x, p=p, p_0=p_0, gamma=gamma,
                     gamma_0=gamma_0, std_y=std_y)

    nn = 10000
    x, t, y, ycf, z, _ = d.sample_joint_data_points(nn)
    print("x:", x)
    print("")
    print("t:", t)
    print("")
    print("y:", y)
    print("")
    print("ycf", ycf)
    print("")
    print("z:", z)
    print("")

    # d.train_q_network(num_iter=num_iter)
    z_values = np.random.normal(size=(5, dim_z))
    p1 = d.get_prob_z(x, t, z_values)
    d._train_q_network(verbose=False)
    p2 = d.get_prob_z(x, t, z_values)
    diffs = np.abs(p1 - p2) / (0.5 * p1 + 0.5 * p2)
    print(diffs.shape)
    print(p1)
    print(p2)
    print("mean diff:", np.mean(diffs.flatten()))
    print("median diff:", np.median(diffs.flatten()))
    print("std diff:", np.std(diffs.flatten()))
    print("max diff:", np.max(diffs.flatten()))

    # print("theoretical cov:", d.x_cov)
    # print("empirical cov:", np.cov(x.T))


if __name__ == "__main__":
    debug()
