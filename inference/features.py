import pandas as pd

from training.features import build_features


def build_features_from_json(data: dict):
    df = pd.DataFrame([data])
    # fill missing numeric with 0
    df = df.fillna(0)
    return build_features(df)