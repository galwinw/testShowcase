from .knn import (
    load_dataset,
    train_test_split,
    predict_classification,
    accuracy_metric,
)

def tune_num_neighbors(train, val, k_values):
    """Return the k in ``k_values`` that performs best on ``val``."""
    best_k = None
    best_acc = -1.0
    actual = [row[-1] for row in val]
    for k in k_values:
        predictions = [predict_classification(train, row, k) for row in val]
        acc = accuracy_metric(actual, predictions)
        if acc > best_acc:
            best_acc = acc
            best_k = k
    return best_k, best_acc


def run_pipeline(data_path="data/iris.csv", k_values=None):
    dataset = load_dataset(data_path)
    train_full, test = train_test_split(dataset, test_ratio=0.2, seed=1)
    train, val = train_test_split(train_full, test_ratio=0.25, seed=1)
    if k_values is None:
        k_values = [1, 3, 5, 7, 9]
    best_k, _ = tune_num_neighbors(train, val, k_values)
    predictions = [predict_classification(train_full, row, best_k) for row in test]
    actual = [row[-1] for row in test]
    accuracy = accuracy_metric(actual, predictions)
    return best_k, accuracy


if __name__ == "__main__":
    best_k, accuracy = run_pipeline()
    print(f"Best k: {best_k}\nAccuracy: {accuracy:.2%}")