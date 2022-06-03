import os
import cv2

def get_small_images():
    batch_path = "./safetyBatches/Batch_0"
    image_sizes = []
    threshold = (30, 30, 3)
    critical_images = []
    for batch, directories, images in os.walk(batch_path):
        for image in images:
            image_path = os.path.join(batch, image)
            single_image = cv2.imread(image_path)
            image_shape = single_image.shape
            image_size = image_shape

            if image_size[1] < threshold[1]:
                critical_images.append(os.path.basename(image))

    print("These images could be prediced false due to their size:\n", critical_images)
#get_small_images()