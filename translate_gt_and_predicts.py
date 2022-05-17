#Created in a team

import numpy as np

def translate_gt_and_predicts(mapping_set, data_set):
    i = 0
    translated_set = np.empty(np.size(data_set), dtype=int)
    for i in range(np.size(data_set)):
        translated_set[i] = np.array([int(mapping_set[data_set[i]])])

    return translated_set
