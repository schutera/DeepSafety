
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def monitor_predictions(prediction_array):
    i = 1
    warning_pic = []                                                                    # declare an array for all critical pictures
    pic_names = []                                                                      # declare an array for all warning picture names 
    for picture in prediction_array:                                                    # go through all picture arrays and always pick one of them
        softmx = tf.nn.softmax(picture)                                                 # transform the array in probabilities sumed to 1
        softmx = softmx.numpy()                                                         # transformed array is to find in numpy
        max_pic = max(softmx)                                                           # function finds the maximum value of the array
        for class_percentage in softmx:                                                 # go through the array so through the single probability values
            dif_percentages = max_pic - class_percentage                                # calculate the difference between the highest and the actual value
            if(0 < dif_percentages <= 0.10):                                            # if the difference is less or equal to ten percent, go into if loop
                print("Warning: There is another possible class for picture %i!"%i)     # throw a warning to the user
                warning_pic.append(softmx)                                              # add the critical picture to the warning array
                if(pic_names != "picture_%i"%i):   
                    pic_names.append("picture_%i"%i)                                    # add the name of the critical picture to names array
        i = i + 1
    j = 0
    old_pictures = glob.glob("warning_pictures/*.png")                                  # find all old histogram in the dedicated folder
    for old_pic in old_pictures:                                
        os.remove(old_pic)                                                              # removing the old histogram from the folder
    print("removed old warning pictures")
    for pic in warning_pic:                                                             # go through all critical pictures
        plt.hist(pic, bins=50, histtype= "stepfilled")                                  # plot a histogram based on one critical pic
        plt.ylabel('number of cases')                                                   # label axes
        plt.xlabel('percentage the picture depends to a class')                         # label axes
        plt.savefig('warning_pictures/%s.png'%pic_names[j])                             # safe the new histogram to the dedicated folder
        plt.close()                                                                     # finalize the plot of this pic
        j = j + 1  
    plt.show()                                                                          # show the plots
    print("save needed warning pictures. They will be removed next time running the program!")
