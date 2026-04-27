import pytest
import pandas as pd

from src.train import train_model
from src.predict import predict


@pytest.fixture(scope="module")
def model():
    """Train a fresh sentiment model on the project dataset."""
    df = pd.read_csv("data/sentiments.csv")
    return train_model(df["text"], df["label"])


@pytest.mark.parametrize(
    "text, expected_label",
    [
        ("Absolutely brilliant and fantastic, I loved it", 1),
        ("Terrible and boring, I would not recommend it", 0),
    ],
)
def test_predict_sanity_checks(model, text, expected_label):
    """The model should classify obvious sentences correctly."""
    result = predict(model, [text])
    assert result[0] == expected_label
