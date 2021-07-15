class Realization:
    def __init__(self, training_set=None, test_set=None, xi_t=None, sigmas=None, weights=None, scores=None):
        self.training_set = training_set
        self.test_set = test_set
        self.xi_t = xi_t
        self.sigmas = sigmas
        self.weights = weights
        self.scores = scores
