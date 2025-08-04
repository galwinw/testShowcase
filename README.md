# testShowcase

This repository demonstrates a simple data science pipeline built with pure Python. It generates a synthetic iris-like dataset and trains a k-nearest neighbors classifier to showcase CODEX capabilities. The pipeline performs a small hyperparameter search to pick the number of neighbors that yields the best validation accuracy.


## Usage

Generate the dataset (already committed):

```
python data/generate_iris.py
```

Run the pipeline:

```
python -m src.pipeline
```

The script reports the chosen ``k`` value and the resulting accuracy on the held-out test set.


```
pytest
```
