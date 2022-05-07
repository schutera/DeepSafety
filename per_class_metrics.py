from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from confidence import *


def per_class_metrics(
    model_output, predictions, test_labels, model_name=None, thres=0.75
):
    """plot the metrics for each class and return as dataframe

    :param predictions: the system prediction
    :type predictions: 1D-array
    :param test_labels: the grund truth
    :type test_labels: 1D-array
    :param model_name: the name of the model, defaults to None
    :type model_name: str, optional
    :return: the metrics for each class
    :rtype: DataFrame
    """

    PREDICTION = "prediction"
    CLASS = "class"
    ACCURACY = "accuracy"
    FALSE_CONFIDENCE = f"false confidence (thres={thres})"
    MISSING_CONFIDENCE = f"missing confidence (thres={thres})"

    # merge predictions and grund truth and build the dataframe
    false_confidence = has_false_confidence(model_output, test_labels, thres)
    missing_confidence = has_missing_confidence(model_output, test_labels, thres)
    df = pd.DataFrame(
        np.vstack((predictions, test_labels, false_confidence, missing_confidence)).T,
        columns=[
            PREDICTION,
            CLASS,
            FALSE_CONFIDENCE,
            MISSING_CONFIDENCE,
        ],
    )
    df[ACCURACY] = df[PREDICTION] == df[CLASS]

    grouped_acc = (
        df[
            [
                CLASS,
                ACCURACY,
                FALSE_CONFIDENCE,
                MISSING_CONFIDENCE,
            ]
        ]
        .groupby(CLASS)
        .mean()
    )

    ax = grouped_acc.plot.bar()

    if model_name is not None:
        ax.set_title(f"Model: {model_name}")

    fig = ax.get_figure()
    fig.savefig(
        "accuracies{}.png".format(f"_{model_name}" if model_name is not None else "")
    )

    return grouped_acc
