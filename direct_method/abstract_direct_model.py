class AbstractDirectModel(object):
    def __init__(self):
        pass

    def train(self, x, t, y, num_treatment):
        raise NotImplementedError()

    def predict_y_t(self, x, t_val):
        raise NotImplementedError()


