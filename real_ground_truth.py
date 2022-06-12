from operator import le
import numpy as np

#this function improves the ground truth to the actual classname rather than the place in the classarray
def set_real_ground_truth (test_labels, class_names):
    ground_truth =[]
    i = 0
    for label in test_labels:
        ground_truth.append(class_names[label])

    return ground_truth

#this function improves the predictions to the actual classname rather than the place in the classarray while still keeping the wrong pridictions
def set_real_prediction(predictions, class_names):
    real_prediction =[]
    lenght = len(class_names)
    for label in predictions:
        if label < lenght:
            real_prediction.append(class_names[label])      #replace with acutal classname
        else:
             real_prediction.append(str(label))             #convert label to string as it is an int right now

    return real_prediction

#this function detects the wrong predictions and saves them in an array
def get_difference(truth, predicition):
    difference = []
    i = 0
    for label in predicition:
        if truth[i] != label:                   #saving the error in the differncearray with 3 spaces per error:
            difference.append(label)            #1. the class it predicts   
            difference.append('should be')      #2. the words 'should be' to read it better if its printed
            difference.append(truth[i])         #3. the class is acutually is
        i = i + 1
    
    return difference

#this function either prints all differences or just the class with the most errors depending on the total percentage of errors
def print_difference(truth, difference):
    number_difference = len(difference)/3           #getting number of errors by diving with 3 because 1 error is saved in 3 spaces
    percentage = number_difference/len(truth) 
    if percentage < 0.1:
        print('Difference: ', difference)
    else:
        error_class = difference[0]                 #initialising all the temp and normal variables
        temp_class = difference[0]                  #initialised with difference[0] so that it is set to the first class
        counter = 0
        temp_counter = 0
        i = 0                                       #i as the varible which space of the array is looked at
        while i < len(difference):                  #staying in the loop as long as the spacenumber is smaller than the length of the array
            if difference[i] == temp_class:         #if they are the same then just add one to temp_counter and go to next error --> add 3 to i
                temp_counter = temp_counter + 1
                i = i + 3
            else:
                if temp_counter > counter:          #check if the temp_counter is higher than counter 
                    counter = temp_counter          #overide all the temp to the actual variables and reset temp to zero or differnence[i]
                    error_class = temp_class
                    temp_class = difference[i]
                    temp_counter = 0
                else:                               #just reset temp variables
                    temp_class = difference[i]
                    temp_counter = 0
        print('Class with most error is: ', error_class, 'with ', counter, 'errors')         #print class with most error and the number of errors   



    
