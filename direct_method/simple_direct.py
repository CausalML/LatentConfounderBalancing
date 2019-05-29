import copy
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data_model.simple_continuous_model import SimpleContinuousModel
from direct_method.abstract_direct_model import AbstractDirectModel


class SimpleDirectModel(AbstractDirectModel):
    def __init__(self):
        AbstractDirectModel.__init__(self)
        self.models = None

    def train(self, x, t, y, num_treatment, num_epochs=500, batch_size=128):
        self.models = []
        num_data = x.shape[0]
        x_dim = x.shape[1]
        for t_val in range(num_treatment):
            model = MuXNetwork(x_dim).double()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            i_t = [i for i in range(num_data) if t[i] == t_val]
            num_train = int(len(i_t) * 0.8)
            num_dev = len(i_t) - num_train

            random.shuffle(i_t)
            train_idx = i_t[:num_train]
            dev_idx = i_t[num_train:]
            x_train = torch.from_numpy(x[train_idx]).double()
            y_train = torch.from_numpy(y[train_idx]).double()
            if num_dev > 0:
                x_dev = torch.from_numpy(x[dev_idx]).double()
                y_dev = torch.from_numpy(y[dev_idx]).double()
                best_model = None
                min_loss = float("inf")
            num_no_progress = 0
            for epoch in range(num_epochs):
                random.shuffle(train_idx)
                i = 0
                while i < num_train:
                    x_batch = x_train[i:i+batch_size]
                    y_batch = y_train[i:i+batch_size]
                    y_batch_pred = torch.squeeze(model(x_batch))
                    loss = ((y_batch_pred - y_batch) ** 2).mean()
                    # print("x_batch:", x_batch[:8])
                    # print("y_batch:", y_batch[:8])
                    # print("y_batch_pred:", y_batch_pred[:8])
                    # print("loss:", loss)
                    # print("")
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    i += batch_size

                if num_dev > 0:
                    y_dev_pred = torch.squeeze(model(x_dev))
                    dev_loss = ((y_dev_pred - y_dev) ** 2).mean()
                    if dev_loss < min_loss:
                        min_loss = float(dev_loss)
                        best_model = copy.deepcopy(model)
                        num_no_progress = 0
                    else:
                        num_no_progress += 1
                        if num_no_progress >= 20:
                            # print("broken at epoch %d with dev loss %f"
                            #       % (epoch, min_loss))
                            break
                    if epoch % 100 == 0:
                        pass
                        # print("epoch %d, dev_loss %f" % (epoch, float(dev_loss)))

            # end of training this t-level, append model to list
            if num_dev > 0:
                self.models.append(best_model)
            else:
                self.models.append(model)

    def predict_y_t(self, x, t_val):
        if self.models is None:
            raise LookupError("no trained model (need to run train first)")
        x = torch.from_numpy(x).double()
        return torch.squeeze(self.models[t_val](x)).detach().numpy()


class MuXNetwork(torch.nn.Module):
    def __init__(self, x_dim):
        torch.nn.Module.__init__(self)
        self.x_dim = x_dim
        self.linear_1 = nn.Linear(x_dim, 100)
        # self.linear_2 = nn.Linear(100, 100)
        self.linear_3 = nn.Linear(100, 1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).double()
        h = F.leaky_relu(self.linear_1(x))
        # h = F.leaky_relu(self.linear_2(h))
        return self.linear_3(h)


def debug():
    num_treatment = 2
    x_dim = 4
    data_model = SimpleContinuousModel()
    num_data = 1000
    x, t, y, y_cf, z, sigma = data_model.sample_joint_data_points(num_data)

    direct_model = SimpleDirectModel()
    direct_model.train(x, t, y, x_dim, num_treatment)
    print("y(0):", y_cf[:10, 0])
    print("pred y(0):", direct_model.predict_y_t(x[:10], t_val=0))
    print("y(1):", y_cf[:10, 1])
    print("pred y(1):", direct_model.predict_y_t(x[:10], t_val=1))



if __name__ == "__main__":
    debug()
