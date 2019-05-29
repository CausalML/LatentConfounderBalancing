

class AbstractWeightsLearning(object):
    def __init__(self, num_data, num_treatment, data_model):
        self.num_data = num_data
        self.num_treatment = num_treatment
        self.data_model = data_model
        self.policy = None

    def update_policy(self, policy):
        # policy is function mapping x value to treatment
        self.policy = policy

    def check_data(self, x, t, y):
        assert(len(x) == self.num_data, "incorrect size of x array")
        assert(len(t) == self.num_data, "incorrect size of t array")
        assert(len(y) == self.num_data, "incorrect size of y array")

    def train(self, x, t, y):
        """
        :param x: array of logged x values
        :param t: array of logged t values
        :param y: array of logged y values
        :return: W: array of weights
        """
        raise NotImplementedError()
