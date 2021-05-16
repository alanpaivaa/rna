class PerceptronNetwork:
    def __init__(self):
        self.one_hot_encodings = None
        self.one_hot_encoding_size = None

    def generate_one_hot_encodings(self, dataset):
        max_class = -1
        for row in dataset:
            max_class = max(max_class, row[-1])
        self.one_hot_encoding_size = max_class + 1

        self.one_hot_encodings = dict()
        for i in range(self.one_hot_encoding_size):
            encoding = [0] * self.one_hot_encoding_size
            encoding[i] = 1
            self.one_hot_encodings[i] = encoding

    def one_hot_encode(self, rows):
        assert self.one_hot_encodings is not None

        for row in rows:
            klass = row[-1]
            encoding = self.one_hot_encodings[klass]
            row.pop()
            for i in encoding:
                row.append(i)

    def train(self, training_set):
        self.generate_one_hot_encodings(training_set)
        self.one_hot_encode(training_set)
