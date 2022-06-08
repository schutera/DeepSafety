from operator import le
import numpy as np

def set_real_ground_truth (test_labels, class_names):
    ground_truth =[]
    i = 0
    for label in test_labels:
        ground_truth.append(class_names[label])

    return ground_truth


def set_real_prediction(predictions, class_names):
    real_prediction =[]
    lenght = len(class_names)
    for label in predictions:
        if label < lenght:
            real_prediction.append(class_names[label])
        else:
             real_prediction.append(str(label))

    return real_prediction

def get_difference(truth, predicition):
    difference = []
    i = 0
    for label in predicition:
        if truth[i] != label:
            difference.append(label)
            difference.append('should be')
            difference.append(truth[i])
        i = i + 1
    
    return difference

def print_difference(truth, difference):
    number_difference = len(difference)/3
    percentage = number_difference/len(truth) 
    if percentage < 0.1:
        print('Difference: ', difference)
    else:
        error_class = difference[0]
        temp_class = difference[0]
        i = 0
        counter = 0
        temp_counter = 0
        while i < len(difference):
            if difference[i] == temp_class:
                temp_counter = temp_counter + 1
                i = i + 3
            else:
                if temp_counter > counter:
                    counter = temp_counter
                    error_class = temp_class
                    temp_class = difference[i]
                    temp_counter = 0
                else: 
                    temp_class = difference[i]
                    temp_counter = 0
        print('Class with most error is: ', error_class, 'with ', counter, 'errors')            



    
