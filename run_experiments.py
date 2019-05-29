import argparse
from collections import defaultdict
from multiprocessing import Queue, Process
import os
import random
import numpy as np
import pandas as pd
import torch

from data_model.simple_continuous_model import SimpleContinuousModel
from direct_method.simple_direct import SimpleDirectModel
from direct_method.two_stage_direct import TwoStageDirectModel
from evaluation.estimate_policy_value import estimate_policy_value
from evaluation.estimate_policy_value_direct import \
    get_mu_t_direct_train_test, get_dr_policy_value
from utils.hide_output import HideOutput
from weights_learning.quadprog_learning_x_continuous import \
    BalancedWeightsLearningContinuousQuadprogX
from weights_learning.quadprog_learning_z_continuous import \
    BalancedWeightsLearningContinuousQuadprog


def y_activation_cubic(y):
    return y ** 3


def y_activation_sign(y):
    return 3 * np.sign(1.0 * y)


def y_activation_exp(y):
    return np.exp(1.0 * y)


def toy_continuous_policy(x, t):
    # score = (x * np.array([-1, 1, -1, 2])).sum() / 10
    score = (x * np.array([-1, 2, 2, -1, -1, -1, 1, 1, 1, -1])).sum() * 0.1
    zero_probability = np.exp(score) / (np.exp(score) + np.exp(-score))
    if t == 0:
        return zero_probability
    else:
        return 1 - zero_probability


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--random_seed", default=527, type=int,
                        help="initial random seed of first process "
                             "(i'th process starts with seed random_seed+i)")
    parser.add_argument("-l", "--link_function", default="step",
                        help="link function to use with data model (available"
                             " options are: step, exp, cubic, linear)",
                        type=str)
    parser.add_argument("-n", "--num_reps", default=64, type=int,
                        help="number of repetitions")
    parser.add_argument("-p", "--num_procs", default=1, type=int,
                        help="number of parallel processes")
    parser.add_argument()
    args = parser.parse_args()

    num_treatment = 2
    if args.link_function == "step":
        y_activation = y_activation_sign
    elif args.link_function == "exp":
        y_activation = y_activation_exp
    elif args.link_function == "cubic":
        y_activation = y_activation_cubic
    elif args.link_function == "linear":
        y_activation = None
    else:
        raise ValueError("Invalid link activation:", args.link_function)

    data_model_class = SimpleContinuousModel
    data_model_args = {"y_activation": y_activation}
    kernel = "rbf"
    policy = toy_continuous_policy
    num_data_policy_estimate = 1000000
    num_data_range = (2000, 1000, 500, 200)
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "%s_results.csv" % args.link_function)

    job_queue = Queue()
    results_queue = Queue()

    num_jobs = 0
    for num_data in num_data_range:
        for rep in range(args.num_reps):
            num_jobs += 1
            job_queue.put((num_data, rep))

    procs = []
    for p_i in range(args.num_proces):
        job_queue.put("STOP")
        config = {
            "num_treatment": num_treatment,
            "data_model_class": data_model_class,
            "data_model_args": data_model_args,
            "kernel": kernel,
            "policy": policy,
            "link_function": args.link_function,
            "seed": args.random_seed + p_i,
        }
        p_args = (job_queue, results_queue, config)
        p = Process(target=worker_loop, args=p_args)
        p.start()
        procs.append(p)

    rows = []
    all_results = defaultdict(lambda: defaultdict(list))
    num_done = 0
    while num_done < num_jobs:
        num_data, rep, results = results_queue.get()
        for method, tau in results.items():
            row = {"method": method, "num_data": num_data, "rep": rep,
                   "tau": tau}
            rows.append(row)
            all_results[num_data][method].append(tau)
        num_done += 1
    for p in procs:
        p.join()

    data_model = data_model_class(**data_model_args)
    policy_value = estimate_policy_value(data_model, policy, num_treatment,
                                         num_data=num_data_policy_estimate)
    print("")
    print("true policy value with %s link function: %f"
          % (args.link_function, policy_value))

    for num_data, results in sorted(all_results.items()):
        print("")
        print("printing results for num-data=%d" % num_data)
        for method, tau_list in sorted(results.items()):
            mse = ((np.array(tau_list) - policy_value) ** 2).mean()
            bias = np.abs(np.array(tau_list).mean() - policy_value)
            bias_std = np.array(tau_list).std(ddof=1)
            bias_std /= (args.num_reps ** 0.5)
            variance = mse - bias ** 2
            mse_se = ((np.array(tau_list) - policy_value) ** 2).std(ddof=1)
            mse_se /= (args.num_reps ** 0.5)
            print("method=%s, mse=%.3f±%.3f, bias=%.3f±%.3f, variance=%.3f"
                  % (method, mse, mse_se, bias, bias_std, variance))
    print("")

    for row in rows:
        row["policy_value"] = policy_value
        row["se"] = (row["tau"] - policy_value) ** 2
    data = pd.DataFrame(rows)
    data.sort_values(["num_data", "method", "rep"])
    data.to_csv(save_path, index=False)
    print("saved results to %s" % save_path)


def worker_loop(job_queue, results_queue, config):
    num_treatment = config["num_treatment"]
    data_model_class = config["data_model_class"]
    data_model_args = config["data_model_args"]
    kernel = config["kernel"]
    policy = config["policy"]
    link_function = config["link_function"]
    seed = config["seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_model = data_model_class(**data_model_args)
    for num_data, rep in iter(job_queue.get, "STOP"):
        print("starting learning with num-data=%d, rep=%d, link-function=%s"
              % (num_data, rep, link_function))

        # get data
        x, t, y, _, z, _ = data_model.sample_joint_data_points(num_data)
        x_train, t_train, y_train, _, z_train, _ = data_model.sample_joint_data_points(1000)

        # setup arrays
        # make policy array
        policy_vec_list = []
        for ti in range(num_treatment):
            policy_vec_list.append(np.array([policy(x_, ti) for x_ in x]))
        policy_array = np.stack(policy_vec_list, axis=1)
        policy_value_dict = {}
        weights_dict = {}
        mu_t_dict = {}

        # do balanced weights learning
        print("sampling Z with num-data=%d, rep=%d, link-function=%s"
              % (num_data, rep, link_function))
        with HideOutput():
            z_sample = data_model.sample_z(
                x, t, num_sample=50, thin=5, chains=1)

        learning = BalancedWeightsLearningContinuousQuadprog(
            num_data, num_treatment, data_model, kernel)
        learning.update_policy(policy)
        for sigma_mul in (0.001, 0.2, 1.0, 5.0):
            sigma = np.ones(num_data) * sigma_mul
            w = learning.train(x, t, y, verbose=True, normalized_weights=False,
                               z_sample=z_sample, sigma=sigma,
                               num_sample=50)
            w_norm = learning.train(x, t, y, normalized_weights=True,
                                    z_sample=z_sample, sigma=sigma,
                                    num_sample=50)
            prefix = "balanced_sig%.1f" % sigma_mul
            policy_value_dict[prefix] = (w * y).mean()
            weights_dict[prefix] = w
            policy_value_dict[prefix + "_norm"] = (w_norm * y).mean()
            weights_dict[prefix + "_norm"] = w_norm

            # do balanced weights learning using X instead of Z
            learning_x = BalancedWeightsLearningContinuousQuadprogX(
                num_data, num_treatment, data_model, kernel)
            learning_x.update_policy(policy)
            # w_x = learning_x.train(x, t, y, normalized_weights=False,
            #                        sigma=sigma)
            w_norm_x = learning_x.train(x, t, y, normalized_weights=True,
                                        sigma=sigma)
            prefix = "x_balanced_sig%.1f" % sigma_mul
            # policy_value_dict[prefix] = (w_x * y).mean()
            # weights_dict[prefix] = w_x
            policy_value_dict[prefix + "_norm"] = (w_norm_x * y).mean()
            weights_dict[prefix + "_norm"] = w_norm_x

        # do naive direct evaluation
        naive_direct = SimpleDirectModel()
        mu_t_naive = get_mu_t_direct_train_test(
            x_test=x, x_train=x_train, t_train=t_train, y_train=y_train,
            direct_model=naive_direct, num_treatment=num_treatment)
        policy_value_dict["direct_naive"] =\
            (mu_t_naive * policy_array).sum(1).mean()
        mu_t_dict["direct_naive"] = mu_t_naive

        # do two-stage direct evaluation
        two_stage_direct = TwoStageDirectModel(data_model)
        with HideOutput():
            z_mode_sample_train = data_model.sample_z_mode(
                x_train, t_train, num_sample=200, chains=1)
            z_mode_sample_test = data_model.sample_z_mode(
                x, t, num_sample=200, chains=1)
        naive_train_args = {"z_mode_sample": z_mode_sample_train}
        naive_predict_args = {"z_mode_sample": z_mode_sample_test}
        mu_t_two_stage = get_mu_t_direct_train_test(
            x_test=x, x_train=x_train, t_train=t_train, y_train=y_train,
            direct_model=two_stage_direct, num_treatment=num_treatment,
            train_args=naive_train_args, predict_args=naive_predict_args)
        policy_value_dict["direct_two_stage"] =\
            (mu_t_two_stage * policy_array).sum(1).mean()
        mu_t_dict["direct_two_stage"] = mu_t_two_stage

        # evaluate using x ips scores
        policy_probs = np.array([policy(x[i], t[i]) for i in range(num_data)])
        w_ips_x = policy_probs / data_model.get_propensity_score_x(x, t)
        if w_ips_x.sum() > 0:
            w_ips_x_norm = w_ips_x / w_ips_x.sum() * num_data
        else:
            w_ips_x_norm = np.ones(num_data)
        policy_value_dict["ips_x"] = (w_ips_x_norm * y).mean()
        weights_dict["ips_x"] = w_ips_x_norm

        # uniform and zero baselines baseline
        policy_value_dict["uniform"] = y.mean()
        policy_value_dict["zero"] = 0.0

        # doubly robust methods
        for weight_method, w in weights_dict.items():
            for direct_method, mu_t in mu_t_dict.items():
                dr_policy_value = get_dr_policy_value(
                    t, y, mu_t, policy_array, w)
                method = "dr:%s:%s" % (weight_method, direct_method)
                policy_value_dict[method] = dr_policy_value

        print("finished job num-data=%d, rep=%d, link-function=%s"
              % (num_data, rep, link_function))
        results_queue.put((num_data, rep, policy_value_dict))


if __name__ == "__main__":
    main()


