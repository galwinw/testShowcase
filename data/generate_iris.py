import csv
import os
import random

def generate_iris(path=None, seed=0):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "iris.csv")
    random.seed(seed)
    rows = []
    for _ in range(50):
        rows.append([
            round(random.gauss(5, 0.4), 2),
            round(random.gauss(3.5, 0.3), 2),
            round(random.gauss(1.4, 0.2), 2),
            round(random.gauss(0.2, 0.1), 2),
            "setosa",
        ])
    for _ in range(50):
        rows.append([
            round(random.gauss(6, 0.5), 2),
            round(random.gauss(2.8, 0.3), 2),
            round(random.gauss(4.5, 0.4), 2),
            round(random.gauss(1.3, 0.2), 2),
            "versicolor",
        ])
    for _ in range(50):
        rows.append([
            round(random.gauss(6.5, 0.6), 2),
            round(random.gauss(3.0, 0.3), 2),
            round(random.gauss(5.5, 0.4), 2),
            round(random.gauss(2.0, 0.2), 2),
            "virginica",
        ])
    random.shuffle(rows)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "species",
        ])
        writer.writerows(rows)

if __name__ == "__main__":
    generate_iris()
