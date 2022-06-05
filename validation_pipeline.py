# This is the seed for your validation pipeline. It will allow you to load a model and run it on data from a directory.

# //////////////////////////////////////// Setup

import numpy as np
import tensorflow as tf
import correct_ground_truth as gr
import get_small_images as gsi
import calc_prediction_probability as prob

# //////////////////////////////////////// Load model

model_name = "1651981229"
import_path = "./tmp/saved_models/{}".format(int(model_name))
model = tf.keras.models.load_model(import_path)

# //////////////////////////////////////// Load data
# You will need to unzip the respective batch folders.
# Obviously Batch_0 is not sufficient for testing as you will soon find out.
data_root = "./safetyBatches/Batch_1/"
batch_size = 32
img_height = 224
img_width = 224

test_ds = tf.keras.utils.image_dataset_from_directory(
  data_root,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  shuffle=False
)

test_data = np.array([0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 5, 6, 7, 8, 9])

# Get information on your classes
class_names = np.array(test_ds.class_names)
print('Classes available: ', class_names)

# get the ground truth labels
test_labels = np.concatenate([y for x, y in test_ds], axis=0)

# Remember that we had some preprocessing before our training this needs to be repeated here
# Preprocessing as the tensorflow hub models expect images as float inputs [0,1]
normalization_layer = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))  # Where x—images, y—labels.

test_labels_n = gr.correct_ground_truth(test_labels, class_names)

# //////////////////////////////////////// Inference.
predictions = model.predict(test_ds)
softmax = tf.nn.softmax(predictions,-1)
predictions = np.argmax(predictions, axis=1)

test_data = gr.correct_ground_truth(predictions, test_data)

print('Predictions: ', test_data)
print('Ground truth: ', test_labels)
print('Corrected Ground truth: ', test_labels_n)


# //////////////////////////////////////// Let the validation begin
# Probably you will want to at least migrate these to another script or class when this grows..
def accuracy(predictions, test_labels):
    metric = tf.keras.metrics.Accuracy()
    metric.update_state(predictions, test_labels)
    return metric.result().numpy()

print('Accuracy: ', accuracy(test_data, test_labels_n))

gsi.get_small_images()
prob.calc_prediction_probability(softmax)

# There is more and this should get you started: https://www.tensorflow.org/api_docs/python/tf/keras/metrics
# However it is not about how many metrics you crank out, it is about whether you find the meangingful ones and report on them.
# Think about a system on how to decide which metric to go for..

# You are looking for a great package to generate your reports, let me recommend https://plotly.com/dash/