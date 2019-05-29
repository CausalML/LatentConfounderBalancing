import numpy as np


def estimate_policy_value_direct_cv(x, t, y, direct_model, policy,
                                    num_treatment, num_cv=5,
                                    train_args=None, predict_args=None):
    if train_args is None:
        train_args = {}
    if predict_args is None:
        predict_args = {}
    num_data = x.shape[0]
    policy_vec_list = []
    mu_vec_list = []
    for i in range(num_cv):
        start_i = (num_data // num_cv) * i
        end_i = (num_data // num_cv) * (i + 1) if i < num_cv - 1 else num_data
        idx_test = [i for i in range(num_data) if start_i <= i < end_i]
        idx_train = [i for i in range(num_data) if i not in idx_test]
        x_train, t_train, y_train = x[idx_train], t[idx_train], y[idx_train]
        x_test, t_test, y_test = x[idx_test], t[idx_test], y[idx_test]
        direct_model.train(x_train, t_train, y_train, num_treatment,
                           **train_args)

        for ti in range(num_treatment):
            policy_vec = np.array([policy(x_, ti) for x_ in x_test])
            mu_vec = direct_model.predict_y_t(x_test, ti, **predict_args)
            policy_vec_list.append(policy_vec)
            mu_vec_list.append(mu_vec)

    policy_vec = np.concatenate(policy_vec_list)
    mu_vec = np.concatenate(mu_vec_list)
    return (policy_vec * mu_vec).mean() * num_treatment



def get_mu_t_direct_train_test(
        x_train, t_train, y_train, x_test, direct_model, num_treatment,
        train_args=None, predict_args=None):
    if train_args is None:
        train_args = {}
    if predict_args is None:
        predict_args = {}

    direct_model.train(x_train, t_train, y_train, num_treatment, **train_args)

    mu_vec_list = []
    for ti in range(num_treatment):
        mu_vec = direct_model.predict_y_t(x_test, ti, **predict_args)
        mu_vec_list.append(mu_vec)
    return np.stack(mu_vec_list, axis=1)


def get_dr_policy_value(t, y, mu_t_array, policy_array, w):
    direct_value = (mu_t_array * policy_array).sum(1).mean()
    mu_t_vec = np.array([mu_t_array[i, t_] for i, t_ in enumerate(t)])
    correction = ((y - mu_t_vec) * w).mean()
    return direct_value + correction
