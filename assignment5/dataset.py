from assignment5.helpers import load_dataset


class Dataset:
    def __init__(self, filename, encoding=None):
        self.filename = filename
        self.encoding = encoding

    def generate_encoding(self, dataset):
        if type(dataset[0][-1]) != str:
            return

        self.encoding = dict()
        encoding_set = set()
        for row in dataset:
            if row[-1] not in encoding_set:
                self.encoding[row[-1]] = len(encoding_set)
                encoding_set.add(row[-1])

    def encoded_classes(self, dataset):
        # For values not contained in the encoding dict, we use the value max + 1
        unknown = max(self.encoding.values()) + 1
        for row in dataset:
            klass = row[-1]  # Last element of the array is the class
            if self.encoding.get(klass) is not None:
                row[-1] = self.encoding[row[-1]]
            else:
                row[-1] = unknown
        return dataset

    @staticmethod
    def set_last_column_int(dataset):
        for row in dataset:
            row[-1] = int(row[-1])

    def load(self):
        dataset = load_dataset(self.filename)

        if self.encoding is None:
            self.generate_encoding(dataset)

        if self.encoding is not None:
            dataset = self.encoded_classes(dataset)
        self.set_last_column_int(dataset)
        return dataset
