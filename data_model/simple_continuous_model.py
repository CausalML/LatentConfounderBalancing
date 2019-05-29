import numpy as np
from data_model.low_rank_model import LowRankModel


class SimpleContinuousModel(LowRankModel):
    def __init__(self, y_activation=None):
        # dim_z = 2
        # dim_x = 4
        # num_t = 2
        # v = np.array([[0.5, -0.5, 1, -1], [-1, 0.5, -0.5, 1]]) * 2.0
        # v_0 = np.zeros(4)
        # std_x = np.ones(4) * 2.0
        # p = np.array([[0.5, -0.5], [-0.5, 0.5]])
        # p_0 = np.zeros(2)
        # gamma = np.array([[1, -0.5], [0.5, 1]])
        # gamma_0 = np.zeros(2)
        # std_y = np.ones(2)
        dim_z = 1
        dim_x = 10
        num_t = 2
        v = np.array([[0.5, -1, -0.5, 1, 2, 0, -1, -0.5, -1.5, 0.5]]) * 2.0
        v_0 = np.zeros(dim_x)
        std_x = np.ones(dim_x) * 4.0
        p = np.array([[0.5, -0.5]]) * 1.0
        p_0 = np.zeros(2)
        gamma = np.array([[1, -0.5]])
        gamma_0 = np.zeros(2)
        std_y = np.ones(2) * 0.01
        LowRankModel.__init__(self, dim_z=dim_z, dim_x=dim_x, num_t=num_t,
                              v=v, v_0=v_0, std_x=std_x, p=p, p_0=p_0,
                              gamma=gamma, gamma_0=gamma_0, std_y=std_y,
                              y_activation=y_activation)


def debug():
    data_model = SimpleContinuousModel()
    num_data = 10
    x, t, _, _, z, _ = data_model.sample_joint_data_points(num_data)
    # standata = {
    #     "N": x.shape[0], "Z": data_model.dim_z, "X": data_model.dim_x,
    #     "T": data_model.num_t, "R": data_model.v.T, "R_0": data_model.v_0,
    #     "std_x": data_model.std_x, "P": data_model.p.T, "P_0": data_model.p_0,
    #     "x": x, "t": t + 1}
    #
    # import pystan
    # sm = pystan.StanModel("continuous_model.stan")
    # fit = sm.sampling(data=standata, iter=1500, warmup=500, thin=2,
    #                   chains=2, verbose=False)
    # out = fit.extract("z")
    # print(out["z"].shape)
    # print("mean", out["z"].mean(0))
    # print("std", out["z"].std(0))
    # print
    # print("z", z)
    # print("abs(z - mean)", np.abs(z - out["z"].mean(0)))
    # print("")


if __name__ == "__main__":
    debug()
