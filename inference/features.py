import pandas as pd

from training.features import build_features


def build_features_from_json(data: dict):
    df = pd.DataFrame([data])
    return build_features(df)