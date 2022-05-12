import numpy as np


def has_high_confidence(model_output, thres=0.75):
    """Check if model has high confidence in its output as the classifier should
    return only one definite class.

    To see if the classification is done with high confidence, we want to look at
    the relative difference between the output class (highest value) and the 2nd
    highest value.
    If the 2nd highest value is lower than threshold * (highest - lowest), it will
    be considered high confidence.

    :param model_output: the whole output of the model for all inputs
    :type model_output: 2D-array
    :param thres: the threshold between 0 and 1, defaults to 0.75
    :type thres: float, optional
    :return: True if the inputs result in high confidence outputs
    :rtype: 1D-array
    """

    # shift and scale the interval so min lies at 0 and max lies at 1
    shape = np.shape(model_output)
    repetitions = shape[1]
    min = np.repeat(np.min(model_output, axis=1), repetitions, axis=0).reshape(shape)
    max = np.repeat(np.max(model_output, axis=1), repetitions, axis=0).reshape(shape)
    diff = max - min

    normalized = (model_output - min) / diff  # division by 0 results in inf, no crash

    if len(normalized) < 2:
        return False

    # sort the results to get 2nd highest value
    second = np.sort(normalized, axis=1)[:, -2]

    # compare to threshold
    return second < thres


def has_false_confidence(model_output, ground_truth, thres=0.75):
    """Check if model has high confidence in its output and its prediction
    is wrong.

    :param system_output: the whole output of the model for all inputs
    :type system_output: 2D-array
    :param ground_truth: the correct output
    :type ground_truth: 1D-array
    :param thres: the threshold between 0 and 1, defaults to 0.75
    :type thres: float, optional
    :return: True if prediction is wrong and has high confidence
    :rtype: 1D-array
    """

    high_conf = has_high_confidence(model_output, thres)

    prediction = np.argmax(model_output, axis=1)

    return (prediction != ground_truth) & high_conf


def has_missing_confidence(model_output, ground_truth, thres=0.75):
    """Check if model has low confidence in its output even though its
    prediction is correct.

    :param system_output: the whole output of the model for all inputs
    :type system_output: 2D-array
    :param ground_truth: the correct output
    :type ground_truth: 1D-array
    :param thres: the threshold between 0 and 1, defaults to 0.75
    :type thres: float, optional
    :return: True if prediction is wrong and has high confidence
    :rtype: 1D-array
    """

    high_conf = has_high_confidence(model_output, thres)

    prediction = np.argmax(model_output, axis=1)

    return (prediction == ground_truth) & ~high_conf
