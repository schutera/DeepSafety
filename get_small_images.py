import os
import cv2
from dataclasses import dataclass


def get_small_images(batch_path, probabilities):
    threshold_shape = (50, 50, 3)
    threshold_probability = 90

    @dataclass
    class critical_image:
        path: ...
        probability: ...
        shape: ...

    critical_images = []
    for batch, directories, images in os.walk(batch_path):
        for image in images:
            image_path = os.path.join(batch, image)
            image_shape = cv2.imread(image_path).shape

            """if probabilities[images.index(image)] < threshold_probability:
                critical_images[images.index(image)].append(critical_image(name=image_path, prediction=probabilities[images.index(image)]))
                #critical_images[1].append(probabilities[images.index(image)])"""
            if image_shape[0] < threshold_shape[0] and image_shape[1] < threshold_shape[1]:
                if probabilities[images.index(image)] < threshold_probability:
                    critical_images.append(critical_image(path=image_path, probability=probabilities[images.index(image)], shape=image_shape))

    return critical_images
#get_small_images()