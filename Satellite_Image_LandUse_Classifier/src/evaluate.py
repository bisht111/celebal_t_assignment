from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


def evaluate_model(y_true, y_pred, class_names):

    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names
    ))


def plot_confusion(y_true, y_pred, class_names):

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(8,8))

    disp.plot(
        cmap="Blues",
        ax=ax,
        xticks_rotation=45
    )

    plt.tight_layout()

    plt.show()