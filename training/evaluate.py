from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


def evaluate_model(y_true, y_pred) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted")
    }

    print("Classification report:")
    print(classification_report(y_true, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    return metrics