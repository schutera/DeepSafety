# This is a script for training a neural network with transfer learning on traffic sign recognition / classification

# The script is an adaptation from:
# https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub

# The data to train and validate the model can be downloaded here:
# https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/download

# The code is part of the applied deep learning lecture at RWU
# Contact: Mark.Schutera@gmail.com or mark.schutera@kit.edu


# //////////////////////////////////////// Setup

import numpy as np
import time

import tensorflow as tf
import tensorflow_hub as hub  # conda install -c conda-forge tensorflow-hub


import datetime
from tensorboard import program


# //////////////////////////////////////// Download Backbone in headless mode
# Meaning these are pure feature extractors

mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
inception_v3 = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"

feature_extractor_model = inception_v3  # @param ["mobilenet_v2", "inception_v3"] choose wisely


# //////////////////////////////////////// Data data data
# The data to train and validate the model can be downloaded here:
# https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/download

# store it to a local folder which you need to define here, for now we only care about the Train data part:
data_root = "./data/Train/"

batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_root,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_root,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# check whether all your classes have been loaded correctly @class_names ['0' '1' '10' '11' '12']
class_names = np.array(train_ds.class_names)
print(class_names)

# Preprocessing as the tensorflow hub models expect images as float inputs [0,1]
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))  # Where x—images, y—labels.
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))     # Where x—images, y—labels.

# Then we set up prefetching will just smooth your data loader pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# //////////////////////////////////////// Preparing the model or heating up the coffee machine

# freeze the feature extractor
feature_extractor_layer = hub.KerasLayer(
    feature_extractor_model,
    input_shape=(img_height, img_width, 3),
    trainable=False)

# attach the head fitting the current traffic sign classification task
# the head is a pure fully connected output layer
num_classes = len(class_names)

model = tf.keras.Sequential([
  feature_extractor_layer,
  tf.keras.layers.Dense(num_classes)
])

# model.summary() (in case you care how the whole thing looks now)


# //////////////////////////////////////// Training or wild hand waving on caffeine

# This starts tensorboard to you can check how your training is progressing
# Helping you with tracking your training, resort to tensorboard, which can be accessed via the browser
tracking_address = "./logs"
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', tracking_address])
url = tb.launch()
print(f"Tensorflow listening on {url}")

# This is stuff you are free to play around with
model.compile(
  optimizer=tf.keras.optimizers.Adam(),

  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1)  # Enable histogram computation for every epoch.

NUM_EPOCHS = 50  # This is probably not enough

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=NUM_EPOCHS,
                    callbacks=tensorboard_callback)

# Save your model for later use. Early enough you should think about a model versioning system
# and which information you will need to link with the model when doing so
t = time.time()

export_path = "./tmp/saved_models/{}".format(int(t))
model.save(export_path)
