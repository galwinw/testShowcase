import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.pipeline import run_pipeline


def test_pipeline_accuracy():
    best_k, accuracy = run_pipeline()
    assert accuracy > 0.6
    assert best_k in {1, 3, 5, 7, 9}
