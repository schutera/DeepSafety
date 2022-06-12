import numpy as np

# get maximum prediction probability values for each image
def calc_prediction_probability(softmax):

    # convert tensor of softmax_list into numpy-array
    softmax_list = softmax.numpy()
    prediction_probabilities = []

    # append list with the highest probability for each image
    for i in range(np.shape(softmax_list)[0]):
        prediction_probabilities.append(np.amax(softmax_list[i]) * 100)

    return prediction_probabilities
