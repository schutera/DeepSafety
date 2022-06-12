import os
import cv2
from dataclasses import dataclass

# calculate the threshold for resolution
def calc_resolution_threshold(batch_path, probabilities, threshold_probability):
    low_probability_image_shapes_x = []
    low_probability_image_shapes_y = []

    #iterate over all images of the validation data
    for root, classes, files in os.walk(batch_path):
        for file in files:
            if probabilities[files.index(file)] < threshold_probability:

                # get shape of image (resolution)
                image_shape = cv2.imread(os.path.join(root, file)).shape
                low_probability_image_shapes_x.append(image_shape[0])
                low_probability_image_shapes_y.append(image_shape[1])

    # check if no images have low prediction probability. If so, return negative value
    if len(low_probability_image_shapes_x) == 0 or len(low_probability_image_shapes_y) == 0:
        return -1, -1
    else:
        # calculate average resolution of low probability images
        average_x = sum(low_probability_image_shapes_x) / len(low_probability_image_shapes_x)
        average_y = sum(low_probability_image_shapes_y) / len(low_probability_image_shapes_y)

    return average_x, average_y


# get images with low resolution
def get_small_images(batch_path, probabilities, predictions, ground_truth):
    threshold_probability = 70

    # Alternative fixed threshold_shape (view report chapter 4)
    #threshold_shape = (32, 32, 3)
    threshold_shape = calc_resolution_threshold(batch_path, probabilities, threshold_probability)

    # check if no images with low prediction probatility were detected. If so, return 0
    if threshold_shape[0] < 0 or threshold_shape[1] < 0:
        return 0
    else:
        threshold_shape = (threshold_shape[0], threshold_shape[1])

    print("Threshold for shape: ", threshold_shape)

    # class for image objects
    @dataclass
    class critical_image:
        path: ...
        file_name: ...
        shape: ...
        probability: ...
        prediction: ...
        ground_truth: ...

    # list for storing critical images
    critical_images = []

    # iterate over validation date
    for batch, directories, images in os.walk(batch_path):
        for image in images:
            image_path = os.path.join(batch, image)
            image_shape = cv2.imread(image_path).shape

            # if resolution is lower than threshold resolution, generate image object and add it to list
            if image_shape[0] < threshold_shape[0] and image_shape[1] < threshold_shape[1]:
                critical_images.append(critical_image(
                    path=image_path,
                    file_name=image,
                    shape=image_shape,
                    probability=probabilities[images.index(image)],
                    prediction=predictions[images.index(image)],
                    ground_truth=ground_truth[images.index(image)]
                    ))

        # Alternative function with fixed threshold_shape (view report chapter 4)
            """ if image_shape[0] < threshold_shape[0] and image_shape[1] < threshold_shape[1]:
                    if probabilities[images.index(image)] < threshold_probability:
                        critical_images.append(critical_image(
                            path=image_path,
                            file_name=image,
                            shape=image_shape,
                            probability=probabilities[images.index(image)],
                            prediction=predictions[images.index(image)],
                            ground_truth=ground_truth[images.index(image)]
                            ))"""

    print("Amount of critical images: ", len(critical_images))
    return critical_images
