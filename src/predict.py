import argparse
from typing import Any

import numpy as np
from joblib import load
from numpy.typing import NDArray


def load_model(model_path: str) -> Any:
    """Load and return a trained classifier."""
    return load(model_path)


def predict(model: Any, texts: list[str]) -> NDArray[np.int64]:
    """Predict labels for one or more input texts."""
    return model.predict(texts)


def main(model_path: str, input_texts: list[str]) -> None:
    """Load the model and print predictions for each input text."""
    model = load_model(model_path)
    preds = predict(model, input_texts)
    for text, label in zip(input_texts, preds):
        print(f"{label}\t{text}")  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/sentiment.joblib")
    parser.add_argument("text", nargs="+", help="One or more texts to score")
    args = parser.parse_args()
    main(model_path=args.model, input_texts=args.text)
