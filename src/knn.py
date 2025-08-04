import csv
import math
import random
from collections import Counter


def load_dataset(path):
    dataset = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            features = [
                float(row["sepal_length"]),
                float(row["sepal_width"]),
                float(row["petal_length"]),
                float(row["petal_width"]),
            ]
            dataset.append(features + [row["species"]])
    return dataset


def train_test_split(dataset, test_ratio=0.2, seed=1):
    random.seed(seed)
    dataset_copy = dataset[:]
    random.shuffle(dataset_copy)
    test_size = int(len(dataset_copy) * test_ratio)
    test = dataset_copy[:test_size]
    train = dataset_copy[test_size:]
    return train, test


def euclidean_distance(row1, row2):
    length = len(row1) - 1
    return math.sqrt(sum((row1[i] - row2[i]) ** 2 for i in range(length)))


def get_neighbors(train, test_row, num_neighbors):
    distances = [(train_row, euclidean_distance(train_row, test_row)) for train_row in train]
    distances.sort(key=lambda tup: tup[1])
    neighbors = [distances[i][0] for i in range(num_neighbors)]
    return neighbors


def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = Counter(output_values).most_common(1)[0][0]
    return prediction


def accuracy_metric(actual, predicted):
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    return correct / len(actual)
