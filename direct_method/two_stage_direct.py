import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_model.simple_continuous_model import SimpleContinuousModel

from direct_method.abstract_direct_model import AbstractDirectModel
from utils.hide_output import HideOutput


class TwoStageDirectModel(AbstractDirectModel):
    def __init__(self, data_model):
        AbstractDirectModel.__init__(self)
        self.data_model = data_model
        self.model = None

    def train(self, x, t, y, num_treatment, z_mode_sample=None,
              num_sample_z=1000, num_epochs=1000, batch_size=128):
        if z_mode_sample is None:
            with HideOutput():
                # z_sample = np.random.randn(x.shape[0], num_sample_z, 2)
                z_mode = self.data_model.sample_z_mode(
                    x, t, num_sample=num_sample_z, chains=1)
        else:
            z_mode = z_mode_sample
        z_dim = z_mode.shape[1]

        self.model = MuZNetwork(z_dim).double()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        y = torch.from_numpy(y).double()
        # y = y.view(-1, 1, 1).repeat(1, num_sample_z, 1).view(-1, 1)
        # z = torch.from_numpy(z_sample).double().contiguous().view(-1, z_dim)
        z = torch.from_numpy(z_mode).double().contiguous()

        train_dev_idx = list(range(z.shape[0]))
        random.shuffle(train_dev_idx)
        num_train = int(len(train_dev_idx) * 0.8)
        num_dev = len(train_dev_idx) - num_train
        train_idx = train_dev_idx[:num_train]
        dev_idx = train_dev_idx[num_train:]
        z_train, y_train = z[train_idx], y[train_idx]
        z_dev, y_dev = z[dev_idx], y[dev_idx]

        if num_dev > 0:
            best_model = None
            min_loss = float("inf")
        num_no_progress = 0
        for epoch in range(num_epochs):
            random.shuffle(train_idx)
            i = 0
            while i < num_train:
                z_batch = z_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                y_batch_pred = torch.squeeze(self.model(z_batch))
                loss = ((y_batch_pred - y_batch) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i += batch_size

            if num_dev > 0:
                y_dev_pred = torch.squeeze(self.model(z_dev))
                dev_loss = ((y_dev_pred - y_dev) ** 2).mean()
                if dev_loss < min_loss:
                    min_loss = float(dev_loss)
                    best_model = copy.deepcopy(self.model)
                    num_no_progress = 0
                else:
                    num_no_progress += 1
                    if num_no_progress >= 20:
                        # print("broken at epoch %d with dev loss %f"
                        #       % (epoch, min_loss))
                        break
                if epoch % 50 == 0:
                    # print("epoch %d, dev_loss %f" % (epoch, float(dev_loss)))
                    pass

        if num_dev > 0:
            self.model = best_model

    def predict_y_t(self, x, t_val, z_mode_sample=None, num_sample_z=1000):
        if self.model is None:
            raise LookupError("no trained model (need to run train first)")

        num_data = x.shape[0]
        t = np.array([t_val]).repeat(num_data)
        if z_mode_sample is None:
            with HideOutput():
                # z_sample = np.random.randn(x.shape[0], num_sample_z, 2)
                z_mode = self.data_model.sample_z_mode(
                    x, t, num_sample=num_sample_z)
        else:
            z_mode = z_mode_sample
        z_dim = z_mode.shape[1]
        # z_flat = torch.from_numpy(z_sample).double().contiguous().view(-1, z_dim)
        # mu_z = self.model(z_flat).view(num_data, num_sample_z)
        # return mu_z.mean(1).detach().numpy()
        z = torch.from_numpy(z_mode).double().contiguous()
        return torch.squeeze(self.model(z)).detach().numpy()


class MuZNetwork(nn.Module):

    def __init__(self, z_dim):
        torch.nn.Module.__init__(self)
        self.z_dim = z_dim
        self.linear_1 = nn.Linear(z_dim, 100)
        # self.linear_2 = nn.Linear(50, 50)
        self.linear_3 = nn.Linear(100, 1)

    def forward(self, z):
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).double()
        h = F.leaky_relu(self.linear_1(z))
        # h = F.leaky_relu(self.linear_2(h))
        return self.linear_3(h)


def debug():
    num_data = 1000
    num_treatment = 2
    y_activation = lambda y_: y_ ** 3
    data_model = SimpleContinuousModel(y_activation=y_activation)
    x, t, y, y_cf, z, _ = data_model.sample_joint_data_points(num_data)
    direct_model = TwoStageDirectModel(data_model)
    direct_model.train(x, t, y, num_treatment,
                       num_sample_z=2000)
    print("y(0):", y_cf[:10, 0])
    print("pred y(0):", direct_model.predict_y_t(x[:10], t_val=0,
                                                 num_sample_z=1000))
    print("y(1):", y_cf[:10, 1])
    print("pred y(1):", direct_model.predict_y_t(x[:10], t_val=1,
                                                 num_sample_z=1000))


if __name__ == "__main__":
    debug()
