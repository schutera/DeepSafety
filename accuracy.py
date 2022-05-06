from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def accuracies(predictions, test_labels, model_name = None):
    """plot the accuracies for each class and return as dataframe

    :param predictions: the system prediction
    :type predictions: 1D-array
    :param test_labels: the grund truth
    :type test_labels: 1D-array
    :param model_name: the name of the model, defaults to None
    :type model_name: str, optional
    :return: the accuracies for each class
    :rtype: DataFrame
    """

    # merge predictions and grund truth and build the dataframe
    df = pd.DataFrame(np.vstack((predictions, test_labels)).T, columns=["prediction", "class"])
    df["accuracy"] = df["prediction"] == df["class"]

    grouped_acc = df[["class", "accuracy"]].groupby("class").mean()
    
    ax = grouped_acc.plot.bar()

    if model_name is not None:
        ax.set_title(f"Model: {model_name}")
        
    fig = ax.get_figure()
    fig.savefig("accuracies{}.png".format(f"_{model_name}" if model_name is not None else ""))

    return grouped_acc