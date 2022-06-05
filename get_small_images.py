import os
import cv2


def get_small_images(batch_path, probabilities):
    threshold = (30, 30, 3)

    critical_images = []
    for i in range(2):
        critical_images.append([])
    for batch, directories, images in os.walk(batch_path):
        for image in images:
            image_path = os.path.join(batch, image)
            image_shape = cv2.imread(image_path).shape

            if probabilities[images.index(image)] < 90:
                critical_images[1].append(probabilities[images.index(image)])
            if image_shape[1] < threshold[1]:
                critical_images[0].append(os.path.basename(image))

    print("These images could be prediced false due to their size:\n", critical_images)
# get_small_images()
