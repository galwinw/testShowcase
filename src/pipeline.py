from .knn import (
    load_dataset,
    train_test_split,
    predict_classification,
    accuracy_metric,
)


def run_pipeline(data_path="data/iris.csv", num_neighbors=5):
    dataset = load_dataset(data_path)
    train, test = train_test_split(dataset, test_ratio=0.2, seed=1)
    predictions = [predict_classification(train, row, num_neighbors) for row in test]
    actual = [row[-1] for row in test]
    accuracy = accuracy_metric(actual, predictions)
    return accuracy


if __name__ == "__main__":
    accuracy = run_pipeline()
    print(f"Accuracy: {accuracy:.2%}")
