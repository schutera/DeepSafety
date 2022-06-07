# This is the seed for your validation pipeline. It will allow you to load a model and run it on data from a directory.

# //////////////////////////////////////// Setup

import numpy as np
import tensorflow as tf
import translate_gt_and_predicts as trl
import evaluation_expansion as evl


# //////////////////////////////////////// Load model
model_name = "1650903961"
import_path = "./tmp/saved_models/{}".format(int(model_name))
model = tf.keras.models.load_model(import_path)

# //////////////////////////////////////// Load data
# You will need to unzip the respective batch folders.
# Obviously Batch_0 is not sufficient for testing as you will soon find out.
data_root = "./safetyBatches/testbatch/"
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

# Testdata mapping vector. selfmade
testdata_mapping = np.array([0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3,
                             30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4, 40, 41, 42, 5, 6, 7, 8, 9])


# Get information on your classes
class_names = np.array(test_ds.class_names)
print('Classes available: ', class_names)
trl.translate_classes(class_names)

# get the ground truth labels
test_labels = np.concatenate([y for x, y in test_ds], axis=0)

# Remember that we had some preprocessing before our training this needs to be repeated here
# Preprocessing as the tensorflow hub models expect images as float inputs [0,1]
normalization_layer = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))  # Where x—images, y—labels.

# //////////////////////////////////////// Inference.
predictions = model.predict(test_ds)
predictions = np.argmax(predictions, axis=1)

# ///////////////////////////////////////// Translation. selfmade

translated_gt = trl.translate_gt_and_predicts(class_names, test_labels)
translated_predicts = trl.translate_gt_and_predicts(testdata_mapping, predictions)

# ///////////////////////////////////////// Confusion Matrix (only suitable for classes 0, 1, 2, 3, 4, 5, 7, 8)
all_cms_data = evl.create_cms(translated_gt, translated_predicts)
evl.calc_precision(all_cms_data)
evl.calc_recall(all_cms_data)
evl.calc_f1(all_cms_data)
evl.export_data_excel(all_cms_data, translated_gt)


# Translated output selfmade
print('Translated Ground truth ', translated_gt)
print('Translated Predictions ', translated_predicts)


# //////////////////////////////////////// Let the validation begin
# Probably you will want to at least migrate these to another script or class when this grows..
def accuracy(translated_predicts, translated_gt):
    metric = tf.keras.metrics.Accuracy()
    metric.update_state(translated_predicts, translated_gt)
    return metric.result().numpy()

print('Accuracy: ', accuracy(translated_predicts, translated_gt))

# There is more and this should get you started: https://www.tensorflow.org/api_docs/python/tf/keras/metrics
# However it is not about how many metrics you crank out, it is about whether you find the meangingful ones and report on them.
# Think about a system on how to decide which metric to go for..

# You are looking for a great package to generate your reports, let me recommend https://plotly.com/dash/
