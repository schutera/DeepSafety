import numpy as np

def correct_ground_truth(test_labels, class_names):
    '''

    :param test_labels:
    :param class_names:
    :return:
    '''
    i = 0
    test_labels_n = np.empty(np.size(test_labels), dtype = int)
    for i in range (np.size(test_labels)):
        test_labels_n[i] = np.array([int(class_names[test_labels[i]])])

    return test_labels_n