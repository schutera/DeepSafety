
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def monitor_predictions(prediction_array, prediction_max):
    i = 1
    warning_pic = []
    pic_names = []
    for picture in prediction_array:
        softmx = tf.nn.softmax(picture)
        softmx = softmx.numpy()
        max_pic = max(softmx)
        for class_percentage in softmx:
            dif_percentages = max_pic - class_percentage
            if(0 < dif_percentages <= 0.15):
                # print("Warning: There is another possible class!")
                warning_pic.append(softmx)
                pic_names.append("picture_%i"%i)
        i = i + 1
    j = 0
    old_pictures = glob.glob("warning_pictures/*.jpg")
    for old_pic in old_pictures:
        os.remove(old_pic)
    print("removed old warning pictures")
    for pic in warning_pic:
        plt.hist(pic, bins=50)  # density=False would make counts
        plt.ylabel('predictions')
        plt.xlabel('number of cases') 
        plt.legend(pic_names) 
        plt.savefig('warning_pictures/%s.jpg'%pic_names[j])
        plt.close()
        j = j + 1  
    plt.show()
    print("save needed warning pictures. They will be removed next time running the program!")

    
    
    # confusion matrix über klassen
