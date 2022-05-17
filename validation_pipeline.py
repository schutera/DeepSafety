# This is the seed for your validation pipeline. It will allow you to load a model and run it on data from a directory.

# //////////////////////////////////////// Setup

import numpy as np
import tensorflow as tf
from vLib.per_class_metrics import *
from vLib.confidence import *

# //////////////////////////////////////// Load model
model_name = "1650909259"
import_path = "./tmp/saved_models/{}".format(int(model_name))
model = tf.keras.models.load_model(import_path)

# //////////////////////////////////////// Load data
# You will need to unzip the respective batch folders.
# Obviously Batch_0 is not sufficient for testing as you will soon find out.
data_root = "./safetyBatches/Batch_0/"
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

# Get information on your classes
class_names = np.array(test_ds.class_names)
print('Classes available: ', class_names)

# get the ground truth labels
test_labels = np.array([]).astype(int)
for t in test_ds:
  test_labels = np.concatenate([test_labels, t[1]])

# Mapping test labels to the folder names instead of the index
for i, test_label in enumerate(test_labels):
    test_labels[i] = int(class_names[test_label])

# Remember that we had some preprocessing before our training this needs to be repeated here
# Preprocessing as the tensorflow hub models expect images as float inputs [0,1]
normalization_layer = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))  # Where x—images, y—labels.

# //////////////////////////////////////// Inference.
output = model.predict(test_ds)
predictions = np.argmax(output, axis=1)
print('Predictions: ', predictions)
print('Ground truth: ', test_labels)


# //////////////////////////////////////// Let the validation begin
# Probably you will want to at least migrate these to another script or class when this grows..
def accuracy(predictions, test_labels):
    metric = tf.keras.metrics.Accuracy()
    metric.update_state(predictions, test_labels)
    return metric.result().numpy()

print('Accuracy: ', accuracy(predictions, test_labels))

print('False confidence: ', np.average(has_false_confidence(output, test_labels)))
print('Low confidence: ', np.average(has_missing_confidence(output, test_labels, 0.1)))

print("Individual metrics:")
print(per_class_metrics(output, predictions, test_labels, model_name, mc_thres=0.1))

# There is more and this should get you started: https://www.tensorflow.org/api_docs/python/tf/keras/metrics
# However it is not about how many metrics you crank out, it is about whether you find the meangingful ones and report on them.
# Think about a system on how to decide which metric to go for..

# You are looking for a great package to generate your reports, let me recommend https://plotly.com/dash/
