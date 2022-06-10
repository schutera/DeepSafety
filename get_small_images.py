import os
import cv2
from dataclasses import dataclass

def calc_resolution_threshold(batch_path, probabilities, threshold_probability):
    low_probability_image_shapes_x = []
    low_probability_image_shapes_y = []
    for root, classes, files in os.walk(batch_path):
        for file in files:
            if probabilities[files.index(file)] < threshold_probability:
                image_shape = cv2.imread(os.path.join(root, file)).shape
                low_probability_image_shapes_x.append(image_shape[0])
                low_probability_image_shapes_y.append(image_shape[1])

    average_x = sum(low_probability_image_shapes_x) / len(low_probability_image_shapes_x)
    average_y = sum(low_probability_image_shapes_y) / len(low_probability_image_shapes_y)

    return average_x, average_y


def get_small_images(batch_path, probabilities, predictions, ground_truth):
    threshold_probability = 70
    threshold_shape = calc_resolution_threshold(batch_path, probabilities, threshold_probability)
    threshold_shape = (threshold_shape[0], threshold_shape[1])
    print("Threshold for shape: ", threshold_shape)

    @dataclass
    class critical_image:
        path: ...
        file_name: ...
        shape: ...
        probability: ...
        prediction: ...
        ground_truth: ...

    critical_images = []

    for batch, directories, images in os.walk(batch_path):
        for image in images:
            image_path = os.path.join(batch, image)
            image_shape = cv2.imread(image_path).shape

            if image_shape[0] < threshold_shape[0] and image_shape[1] < threshold_shape[1]:
                if probabilities[images.index(image)] < threshold_probability:
                    critical_images.append(critical_image(
                        path=image_path,
                        file_name=image,
                        shape=image_shape,
                        probability=probabilities[images.index(image)],
                        prediction=predictions[images.index(image)],
                        ground_truth=ground_truth[images.index(image)]
                        ))

    return critical_images
