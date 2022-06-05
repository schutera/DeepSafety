import numpy as np


def calc_prediction_probability(softmax):
    softmax_list = softmax.numpy()
    prediction_probabilities = []

    for i in range(np.shape(softmax_list)[0]):
        prediction_probabilities.append(np.amax(softmax_list[i]) * 100)

    return prediction_probabilities
