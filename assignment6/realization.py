class Realization:
    def __init__(self, training_set=None, test_set=None, layers=None, scores=None, errors=None):
        self.training_set = training_set
        self.test_set = test_set
        self.layers = layers
        self.scores = scores
        self.errors = errors
