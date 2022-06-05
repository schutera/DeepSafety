import numpy as np

def calc_prediction_probability(softmax):
    prediction_probabilities = []
    for i in softmax:
        prediction_probabilities[i] = np.amin(softmax[i])

    print(prediction_probabilities)