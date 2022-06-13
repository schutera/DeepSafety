import os
import random
import numpy as np
from PIL import Image, ImageStat, ImageEnhance


# returns a 2D array with the average brightness of each image found in directory
def brightness(directory):
    # iterate through files in directory
    brightnesses: object = [[], []]
    i = 0
    for filename in os.listdir(directory):
        im_file = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(im_file):
            # calculate rms brightness#
            filepath = directory + filename
            im = Image.open(im_file).convert('L')
            image_stat = ImageStat.Stat(im)
            brightnesses[0].append([i])
            brightnesses[1].append([image_stat.mean[0]])
            i += 1
    return brightnesses


# augmentation method that creates randomized darkened images in a specific brightness range from the images only for
# images from class "00". This is done due to my computing power. For Batch 3 only class "00" is needed.
def dark_augmentation_short(data_root):
    if not os.path.exists(data_root + "00/00augmentation"):
        os.makedirs(data_root + "00/00augmentation")
        for filename in os.listdir(data_root + "00"):
            filepath = data_root + "00/" + filename
            if not ("00augmentation" in filepath):
                for file in os.listdir(data_root + "00"):
                    if not ("00augmentation" in file):
                        l = Image.open(data_root + "00/" + file)
                        enhancer = ImageEnhance.Brightness(l)
                        l = enhancer.enhance(random.uniform(0.005, 0.25))
                        l.save(data_root + "00/00augmentation/" + file)


# augmentation method that creates randomized darkened images in a specific brightness range from the images of all
# classes found in data_root subdirectories
def dark_augmentation(data_root):
    for directory in os.listdir(data_root):
        if not os.path.exists(data_root + directory + "/" + directory + "augmentation"):
            os.makedirs(data_root + directory + "/" + directory + "augmentation")  #
        if not ("augmentation" in data_root + directory):
            for filename in os.listdir(data_root + directory):
                filepath = data_root + directory + "/" + filename
                if not (directory + "augmentation" in filepath):
                    for file in os.listdir(data_root + directory):
                        if not (directory + "augmentation" in file):
                            l = Image.open(data_root + directory + "/" + file)
                            enhancer = ImageEnhance.Brightness(l)
                            l = enhancer.enhance(random.uniform(0.005, 0.25))  # 0.05
                            l.save(data_root + directory + "/" + directory + "augmentation" + "/" + file)
                            return


# calculates and prints the average brightnesses for training data, darkened training data and validation data
def average_brightnesses():
    batch3_brightnesses = brightness("./safetyBatches/Batch_3/00/")
    training_00_brightnesses = brightness("./data/Train/00/")
    training_00augmentation_brightnesses = brightness("./data/Train/00/00augmentation/")
    print("average of batch3:", np.mean(batch3_brightnesses, 1, int))
    print("average of training (00):", np.mean(training_00_brightnesses, 1, int))
    print("average of training (00_augmentation):", np.mean(training_00augmentation_brightnesses, 1, int))
