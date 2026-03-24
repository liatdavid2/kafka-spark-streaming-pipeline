# explain.py

from feature_mapping import FEATURE_NAMES


def describe_value(feature, value, train_stats):
    """
    Classify feature value as low / normal / high
    based on training mean.
    """
    stats = train_stats.get(feature, {})
    mean = stats.get("mean")

    if mean is None:
        return "unknown"

    if value < mean * 0.7:
        return "low"
    elif value > mean * 1.3:
        return "high"
    else:
        return "normal"


def build_feature_sentence(feature, value, impact):
    """
    Convert feature + SHAP impact into a human-readable sentence.
    """
    name = FEATURE_NAMES.get(feature, feature)

    if impact > 0:
        effect = "increases risk"
    else:
        effect = "reduces risk"

    return f"{name} is {round(value, 2)} and {effect}"