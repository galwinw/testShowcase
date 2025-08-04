import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.pipeline import run_pipeline


def test_pipeline_accuracy():
    accuracy = run_pipeline()
    assert accuracy > 0.6
