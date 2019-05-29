import numpy as np
from data_model.simple_continuous_model import SimpleContinuousModel


def toy_continuous_policy(x, t):
    # score = (x * np.array([-1, 1, -1, 1])).sum()
    score = (x * np.array([-1, 1, -1, 2])).sum()
    zero_probability = np.exp(score) / (np.exp(score) + np.exp(-score))
    if t == 0:
        return zero_probability
    else:
        return 1 - zero_probability


def estimate_policy_value(data_model, policy, num_treatments,
                          num_data=100000):
    x, t, y, _, z, sigma = data_model.sample_joint_data_points(num_data)
    policy_value = 0
    for ti in range(num_treatments):
        policy_vec = np.array([policy(x_, ti) for x_ in x])
        mu_vec = np.array([data_model.mu(z_, ti) for z_ in z])
        policy_value += (policy_vec * mu_vec).mean()
    return policy_value


if __name__ == "__main__":
    y_activation = lambda y_: np.abs(y_) ** 2 / np.sign(y_)
    data_model = SimpleContinuousModel(y_activation=None)
    policy = toy_continuous_policy
    for _ in range(10):
        policy_value = estimate_policy_value(
            data_model=data_model, policy=policy, num_data=1000000,
            num_treatments=2)
        print("policy value:", policy_value)
    print("")
