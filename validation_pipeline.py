# This is the seed for your validation pipeline. It will allow you to load a model and run it on data from a directory.

# //////////////////////////////////////// Setup

import numpy as np
import tensorflow as tf


# //////////////////////////////////////// Load model
model_name = "1654280992"
import_path = "./tmp/saved_models/{}".format(int(model_name))
model = tf.keras.models.load_model(import_path)

# //////////////////////////////////////// Load data
# You will need to unzip the respective batch folders.
# Obviously Batch_0 is not sufficient for testing as you will soon find out.
data_root = "./safetyBatches/Batch_5/"
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
test_labels = np.concatenate([y for x, y in test_ds], axis=0)

# Remember that we had some preprocessing before our training this needs to be repeated here
# Preprocessing as the tensorflow hub models expect images as float inputs [0,1]
normalization_layer = tf.keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))  # Where x—images, y—labels.


# //////////////////////////////////////// Inference.
predictions = model.predict(test_ds)
predictions = np.argmax(predictions, axis=1)
print('Predictions: ', predictions)
print('Ground truth: ', test_labels)


# //////////////////////////////////////// Let the validation begin
# Probably you will want to at least migrate these to another script or class when this grows..
def accuracy(predictions, test_labels):
    metric = tf.keras.metrics.Accuracy()
    metric.update_state(predictions, test_labels)
    return metric.result().numpy()




#Funktion für die Berechnung der Metriken zu Klasse 0, übertragbar für alle Klassen
def MetricsClass0(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:                     #Iteration durch die Arrays Predicition & test_labels 
    if predictions[i] == 0 and test_labels[i] == 0:   #Falls Prediction und test_labels übereinstimmen wird Wert für True Positive erhöht 
      TruePositives = TruePositives + 1
    if predictions[i] == 0 and test_labels[i] != 0 :  #Falls Prediction und test_labels nicht übereinstimmen obwohl 0 predicted wurde, wird Wert für False Positive erhöht
      FalsePositives = FalsePositives +1
    if predictions[i] != 0 and test_labels[i] == 0:  #Falls Prediction und test_labels nicht übereinstimmen und falschlicherweise nicht der richtige Wert predicted wurde, wird Wert für False Negative erhöht
      FalseNegatives += 1
    i = i+1 
  if TruePositives >= 1:            #Berechnung der Metriken nur möglich wenn es mind. einen TruePositve-Wert gibt.
    Precision = (TruePositives/(TruePositives + FalsePositives))    #Berechnung Precision
    Recall = (TruePositives/(TruePositives + FalseNegatives))       #Berechnung Recall
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))     #Berechnung F1-Score
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), round(Recall,3), round(F1_score, 5))    #Rüchgabewerte der Funktion
  else:             #Fall für keinen TruePositive-Wert
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)  #Rückgabe von nur Nullen 



def MetricsClass1(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] == 1 and test_labels[i] == 1:
      TruePositives = TruePositives + 1
    if predictions[i] == 1 and test_labels[i] != 1 :
      FalsePositives = FalsePositives +1
    if predictions[i] != 1 and test_labels[i] == 1:
      FalseNegatives += 1
    i = i+1 
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), round(Recall,3), round(F1_score, 5))
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)


def MetricsClass2(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] == 2 and test_labels[i] == 2:
      TruePositives = TruePositives + 1
    if predictions[i] == 2 and test_labels[i] != 2 :
      FalsePositives = FalsePositives +1
    if predictions[i] != 2 and test_labels[i] == 2:
      FalseNegatives += 1
    i = i+1 
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), round(Recall,3), round(F1_score, 5))
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)


def MetricsClass3(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  3 and test_labels[i] == 3:
      TruePositives += 1
    if predictions[i] ==  3 and test_labels[i] != 3:
      FalsePositives += 1
    if predictions[i] != 3 and test_labels[i] == 3:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 3)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), round(Recall,3), F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), round(Recall,3), round(F1_score, 5))

def MetricsClass4(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  4 and test_labels[i] == 4:
      TruePositives += 1
    if predictions[i] ==  4 and test_labels[i] != 4:
      FalsePositives += 1
    if predictions[i] != 4 and test_labels[i] == 4:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), round(Recall,3), round(F1_score, 5))
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass5(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  5 and test_labels[i] == 5:
      TruePositives += 1
    if predictions[i] ==  5 and test_labels[i] != 5:
      FalsePositives += 1
    if predictions[i] != 5 and test_labels[i] == 5:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), round(Recall,3), round(F1_score, 5))
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)
  
def MetricsClass6(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  6 and test_labels[i] == 6:
      TruePositives += 1
    if predictions[i] ==  6 and test_labels[i] != 6:
      FalsePositives += 1
    if predictions[i] != 6 and test_labels[i] == 6:
      FalseNegatives += 1
    i = i+1
  if TruePositives:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), round(Recall,3), round(F1_score, 5))
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass7(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  7 and test_labels[i] == 7:
      TruePositives += 1
    if predictions[i] ==  7 and test_labels[i] != 7:
      FalsePositives += 1
    if predictions[i] != 7 and test_labels[i] == 7:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), round(Recall,3), round(F1_score, 5))
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)
  
def MetricsClass8(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  8 and test_labels[i] == 8:
      TruePositives += 1
    if predictions[i] ==  8 and test_labels[i] != 8:
      FalsePositives += 1
    if predictions[i] != 8 and test_labels[i] == 8:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), round(Recall,3), round(F1_score, 5))
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass9(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  9 and test_labels[i] == 9:
      TruePositives += 1
    if predictions[i] ==  9 and test_labels[i] != 9:
      FalsePositives += 1
    if predictions[i] != 9 and test_labels[i] == 9:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), round(Recall,3), round(F1_score, 5))
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass10(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  10 and test_labels[i] == 10:
      TruePositives += 1
    if predictions[i] ==  10 and test_labels[i] != 10:
      FalsePositives += 1
    if predictions[i] != 10 and test_labels[i] == 10:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), round(Recall,3), round(F1_score, 5))
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)
      
def MetricsClass11(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  11 and test_labels[i] == 11:
      TruePositives += 1
    if predictions[i] ==  11 and test_labels[i] != 11:
      FalsePositives += 1
    if predictions[i] != 11 and test_labels[i] == 11:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass12(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  12 and test_labels[i] == 12:
      TruePositives += 1
    if predictions[i] ==  12 and test_labels[i] != 12:
      FalsePositives += 1
    if predictions[i] != 12 and test_labels[i] == 12:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)
  
def MetricsClass13(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  13 and test_labels[i] == 13:
      TruePositives += 1
    if predictions[i] ==  13 and test_labels[i] != 13:
      FalsePositives += 1
    if predictions[i] != 13 and test_labels[i] == 13:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)
  
def MetricsClass14(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  14 and test_labels[i] == 14:
      TruePositives += 1
    if predictions[i] ==  14 and test_labels[i] != 14:
      FalsePositives += 1
    if predictions[i] != 14 and test_labels[i] == 14:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass15(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  15 and test_labels[i] == 15:
      TruePositives += 1
    if predictions[i] ==  15 and test_labels[i] != 15:
      FalsePositives += 1
    if predictions[i] != 15 and test_labels[i] == 15:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass16(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  16 and test_labels[i] == 16:
      TruePositives += 1
    if predictions[i] ==  16 and test_labels[i] != 16:
      FalsePositives += 1
    if predictions[i] != 16 and test_labels[i] == 16:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass17(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  17 and test_labels[i] == 17:
      TruePositives += 1
    if predictions[i] ==  17 and test_labels[i] != 17:
      FalsePositives += 1
    if predictions[i] != 17 and test_labels[i] == 17:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass18(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  18 and test_labels[i] == 18:
      TruePositives += 1
    if predictions[i] ==  18 and test_labels[i] != 18:
      FalsePositives += 1
    if predictions[i] != 18 and test_labels[i] == 18:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass19(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  19 and test_labels[i] == 19:
      TruePositives += 1
    if predictions[i] ==  19 and test_labels[i] != 19:
      FalsePositives += 1
    if predictions[i] != 19 and test_labels[i] == 19:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass20(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  20 and test_labels[i] == 20:
      TruePositives += 1
    if predictions[i] ==  20 and test_labels[i] != 20:
      FalsePositives += 1
    if predictions[i] != 20 and test_labels[i] == 20:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass21(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  21 and test_labels[i] == 21:
      TruePositives += 1
    if predictions[i] ==  21 and test_labels[i] != 21:
      FalsePositives += 1
    if predictions[i] != 21 and test_labels[i] == 21:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass22(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  22 and test_labels[i] == 22:
      TruePositives += 1
    if predictions[i] ==  22 and test_labels[i] != 22:
      FalsePositives += 1
    if predictions[i] != 22 and test_labels[i] == 22:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass23(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  23 and test_labels[i] == 23:
      TruePositives += 1
    if predictions[i] ==  23 and test_labels[i] != 23:
      FalsePositives += 1
    if predictions[i] != 23 and test_labels[i] == 23:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass24(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  24 and test_labels[i] == 24:
      TruePositives += 1
    if predictions[i] ==  24 and test_labels[i] != 24:
      FalsePositives += 1
    if predictions[i] != 24 and test_labels[i] == 24:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass25(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  25 and test_labels[i] == 25:
      TruePositives += 1
    if predictions[i] ==  25 and test_labels[i] != 25:
      FalsePositives += 1
    if predictions[i] != 25 and test_labels[i] == 25:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass26(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  26 and test_labels[i] == 26:
      TruePositives += 1
    if predictions[i] ==  26 and test_labels[i] != 26:
      FalsePositives += 1
    if predictions[i] != 26 and test_labels[i] == 26:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass27(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  27 and test_labels[i] == 27:
      TruePositives += 1
    if predictions[i] ==  27 and test_labels[i] != 27:
      FalsePositives += 1
    if predictions[i] != 27 and test_labels[i] == 27:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass28(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  28 and test_labels[i] == 28:
      TruePositives += 1
    if predictions[i] ==  28 and test_labels[i] != 28:
      FalsePositives += 1
    if predictions[i] != 28 and test_labels[i] == 28:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass29(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  29 and test_labels[i] == 29:
      TruePositives += 1
    if predictions[i] ==  29 and test_labels[i] != 29:
      FalsePositives += 1
    if predictions[i] != 29 and test_labels[i] == 29:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass30(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  30 and test_labels[i] == 30:
      TruePositives += 1
    if predictions[i] ==  30 and test_labels[i] != 30:
      FalsePositives += 1
    if predictions[i] != 30 and test_labels[i] == 30:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass31(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  31 and test_labels[i] == 31:
      TruePositives += 1
    if predictions[i] ==  31 and test_labels[i] != 31:
      FalsePositives += 1
    if predictions[i] != 31 and test_labels[i] == 31:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass32(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  32 and test_labels[i] == 32:
      TruePositives += 1
    if predictions[i] ==  32 and test_labels[i] != 32:
      FalsePositives += 1
    if predictions[i] != 32 and test_labels[i] == 32:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass33(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  33 and test_labels[i] == 33:
      TruePositives += 1
    if predictions[i] ==  33 and test_labels[i] != 33:
      FalsePositives += 1
    if predictions[i] != 33 and test_labels[i] == 33:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass34(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  34 and test_labels[i] == 34:
      TruePositives += 1
    if predictions[i] ==  34 and test_labels[i] != 34:
      FalsePositives += 1
    if predictions[i] != 34 and test_labels[i] == 34:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass35(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  35 and test_labels[i] == 35:
      TruePositives += 1
    if predictions[i] ==  35 and test_labels[i] != 35:
      FalsePositives += 1
    if predictions[i] != 35 and test_labels[i] == 35:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)

def MetricsClass36(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  36 and test_labels[i] == 36:
      TruePositives += 1
    if predictions[i] ==  36 and test_labels[i] != 36:
      FalsePositives += 1
    if predictions[i] != 36 and test_labels[i] == 36:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)

def MetricsClass37(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  37 and test_labels[i] == 37:
      TruePositives += 1
    if predictions[i] ==  37 and test_labels[i] != 37:
      FalsePositives += 1
    if predictions[i] != 37 and test_labels[i] == 37:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass38(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  38 and test_labels[i] == 38:
      TruePositives += 1
    if predictions[i] ==  38 and test_labels[i] != 38:
      FalsePositives += 1
    if predictions[i] != 38 and test_labels[i] == 38:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass39(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  39 and test_labels[i] == 39:
      TruePositives += 1
    if predictions[i] ==  39 and test_labels[i] != 39:
      FalsePositives += 1
    if predictions[i] != 39 and test_labels[i] == 39:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass40(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  40 and test_labels[i] == 40:
      TruePositives += 1
    if predictions[i] ==  40 and test_labels[i] != 40:
      FalsePositives += 1
    if predictions[i] != 40 and test_labels[i] == 40:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)

def MetricsClass41(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  41 and test_labels[i] == 41:
      TruePositives += 1
    if predictions[i] ==  41 and test_labels[i] != 41:
      FalsePositives += 1
    if predictions[i] != 41 and test_labels[i] == 41:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

def MetricsClass42(predictions, test_labels):
  i = 0
  TruePositives = 0
  FalsePositives = 0
  FalseNegatives = 0
  while i < predictions.size:
    if predictions[i] ==  42 and test_labels[i] == 42:
      TruePositives += 1
    if predictions[i] ==  42 and test_labels[i] != 42:
      FalsePositives += 1
    if predictions[i] != 42 and test_labels[i] == 42:
      FalseNegatives += 1
    i = i+1
  if TruePositives >= 1:
    Precision = (TruePositives/(TruePositives + FalsePositives))
    Recall = (TruePositives/(TruePositives + FalseNegatives))
    F1_score= (2 * ((Precision * Recall)/(Precision + Recall)))
    F1_score = round(F1_score, 5)
    return(TruePositives, FalsePositives, FalseNegatives, round(Precision,3), Recall, F1_score)
  else:
    Precision = TruePositives
    Recall = TruePositives
    F1_score= TruePositives
    F1_score = TruePositives
    return(TruePositives, FalsePositives, FalseNegatives, Precision, Recall, F1_score)

#Aufruf der Fuktion und Speicherung der Rückgabewerte
metrics42 = MetricsClass42(predictions, test_labels)
metrics41 = MetricsClass41(predictions, test_labels)
metrics40 = MetricsClass40(predictions, test_labels)
metrics39 = MetricsClass39(predictions, test_labels)
metrics38 = MetricsClass38(predictions, test_labels)
metrics37 = MetricsClass37(predictions, test_labels)
metrics36 = MetricsClass36(predictions, test_labels)
metrics35 = MetricsClass35(predictions, test_labels)
metrics34 = MetricsClass34(predictions, test_labels)
metrics33 = MetricsClass33(predictions, test_labels)
metrics32 = MetricsClass32(predictions, test_labels)
metrics31 = MetricsClass31(predictions, test_labels)
metrics30 = MetricsClass30(predictions, test_labels)
metrics29 = MetricsClass29(predictions, test_labels)
metrics28 = MetricsClass28(predictions, test_labels)
metrics27 = MetricsClass27(predictions, test_labels)
metrics26 = MetricsClass26(predictions, test_labels)
metrics25 = MetricsClass25(predictions, test_labels)
metrics24 = MetricsClass24(predictions, test_labels)
metrics23 = MetricsClass23(predictions, test_labels)
metrics22 = MetricsClass22(predictions, test_labels)
metrics21 = MetricsClass21(predictions, test_labels)
metrics20 = MetricsClass20(predictions, test_labels)
metrics19 = MetricsClass19(predictions, test_labels)
metrics18 = MetricsClass18(predictions, test_labels)
metrics17 = MetricsClass17(predictions, test_labels)
metrics16 = MetricsClass16(predictions, test_labels)
metrics15 = MetricsClass15(predictions, test_labels)
metrics14 = MetricsClass14(predictions, test_labels)
metrics13 = MetricsClass13(predictions, test_labels)
metrics12 = MetricsClass12(predictions, test_labels)
metrics11 = MetricsClass11(predictions, test_labels)
metrics10 = MetricsClass10(predictions, test_labels)
metrics9 = MetricsClass9(predictions, test_labels)
metrics8 = MetricsClass8(predictions, test_labels)
metrics7 = MetricsClass7(predictions, test_labels)
metrics6 = MetricsClass6(predictions, test_labels)
metrics5 = MetricsClass5(predictions, test_labels)
metrics4 = MetricsClass4(predictions, test_labels)
metrics3 = MetricsClass3(predictions, test_labels)
metrics2 = MetricsClass2(predictions, test_labels)
metrics1 = MetricsClass1(predictions, test_labels)
metrics0 = MetricsClass0(predictions, test_labels)

#MicroF1-Berechnung aus allen TruePositive, FalsePositive, FalseNegative
def microF1(metrics42, metrics41, metrics40, metrics39, metrics38, metrics37, metrics36, metrics35, metrics34, metrics33, metrics32, metrics31, metrics30, metrics29, metrics28, metrics27, metrics26, metrics25, metrics24, metrics23, metrics22, metrics21, metrics20, metrics19, metrics18, metrics17, metrics16, metrics15, metrics14, metrics13, metrics12, metrics11, metrics10, metrics9, metrics8, metrics7, metrics6, metrics5, metrics4, metrics3, metrics2, metrics1, metrics0):
  allTruePositives = metrics42[0] + metrics41[0] + metrics40[0] + metrics39[0] + metrics38[0] + metrics37[0] + metrics36[0] + metrics35[0] + metrics34[0] + metrics33[0] + metrics32[0] + metrics31[0] + metrics30[0] + metrics29[0] + metrics28[0] + metrics27[0] + metrics26[0] + metrics25[0] + metrics24[0] + metrics23[0] + metrics22[0] + metrics21[0] + metrics20[0] + metrics19[0] + metrics18[0] + metrics17[0] + metrics16[0] + metrics15[0] + metrics14[0] + metrics13[0] + metrics12[0] + metrics11[0] + metrics10[0] + metrics9[0] + metrics8[0] + metrics7[0] + metrics6[0] + metrics5[0] + metrics4[0] + metrics3[0] + metrics2[0] + metrics1[0] + metrics0[0]
  allFalsePositives = metrics42[1] + metrics41[1] + metrics40[1] + metrics39[1] + metrics38[1] + metrics37[1] + metrics36[1] + metrics35[1] + metrics34[1] + metrics33[1] + metrics32[1] + metrics31[1] + metrics30[1] + metrics29[1] + metrics28[1] + metrics27[1] + metrics26[1] + metrics25[1] + metrics24[1] + metrics23[1] + metrics22[1] + metrics21[1] + metrics20[1] + metrics19[1] + metrics18[1] + metrics17[1] + metrics16[1] + metrics15[1] + metrics14[1] + metrics13[1] + metrics12[1] + metrics11[1] + metrics10[1] + metrics9[1] + metrics8[1] + metrics7[1] + metrics6[1] + metrics5[1] + metrics4[1] + metrics3[1] + metrics2[1] + metrics1[1] + metrics0[1] 
  allFalseNegatives = metrics42[2] + metrics41[2] + metrics40[2] + metrics39[2] + metrics38[2] + metrics37[2] + metrics36[2] + metrics35[2] + metrics34[2] + metrics33[2] + metrics32[2] + metrics31[2] + metrics30[2] + metrics29[2] + metrics28[2] + metrics27[2] + metrics26[2] + metrics25[2] + metrics24[2] + metrics23[2] + metrics22[2] + metrics21[2] + metrics20[2] + metrics19[2] + metrics18[2] + metrics17[2] + metrics16[2] + metrics15[2] + metrics14[2] + metrics13[2] + metrics12[2] + metrics11[2] + metrics10[2] + metrics9[2] + metrics8[2] + metrics7[2] + metrics6[2] + metrics5[2] + metrics4[2] + metrics3[2] + metrics2[2] + metrics1[2] + metrics0[2]
  globalPrecision = (allTruePositives/(allTruePositives + allFalsePositives))
  globalRecall = (allTruePositives/(allTruePositives + allFalseNegatives))
  microF1 = (2 * ((globalPrecision * globalRecall)/(globalPrecision + globalRecall)))   #Exakte Berechnung MicroF1 aus globaler Precision und globalem Recall
  return(round(microF1, 4)) #Rückgabewert der Funktion

#MacroF1-Berechnung aus den F1-Scores aller Klassen
def macroF1(metrics42,metrics41,metrics40,metrics39, metrics38, metrics37, metrics36, metrics35, metrics34, metrics33, metrics32, metrics31, metrics30, metrics29, metrics28, metrics27, metrics26, metrics25, metrics24, metrics23, metrics22, metrics21, metrics20, metrics19, metrics18, metrics17, metrics16, metrics15, metrics14, metrics13, metrics12, metrics11, metrics10, metrics9, metrics8, metrics7, metrics6, metrics5, metrics4, metrics3, metrics2, metrics1, metrics0):
  Macro_F1 = (metrics0[5] + metrics1[5] + metrics2[5] + metrics3[5] + metrics5[5] + metrics4[5] + metrics6[5] + metrics7[5] + metrics8[5] + metrics9[5] + metrics10[5] + metrics11[5] + metrics12[5] + metrics13[5] + metrics14[5] + metrics15[5] + metrics16[5] + metrics17[5] + metrics18[5] + metrics19[5] + metrics20[5] + metrics21[5] + metrics22[5] + metrics23[5] + metrics24[5] + metrics25[5] + metrics26[5] + metrics27[5] + metrics28[5] + metrics29[5] + metrics30[5] + metrics31[5] + metrics32[5] + metrics33[5] + metrics34[5] + metrics35[5] + metrics36[5] + metrics37[5] + metrics38[5] + metrics39[5] + metrics40[5] + metrics41[5] + metrics42[5] )/43
  return (round(Macro_F1, 4))

#gewichteter F1 Score nach Anzahl der Beispiele die pro Klasse in dem Safety_Batch sind, macht in diesem Fall kein Sinn, da alle Beispiele gleich oft drin sind
def weightedF1score(metrics42,metrics41,metrics40,metrics39, metrics38, metrics37, metrics36, metrics35, metrics34, metrics33, metrics32, metrics31, metrics30, metrics29, metrics28, metrics27, metrics26, metrics25, metrics24, metrics23, metrics22, metrics21, metrics20, metrics19, metrics18, metrics17, metrics16, metrics15, metrics14, metrics13, metrics12, metrics11, metrics10, metrics9, metrics8, metrics7, metrics6, metrics5, metrics4, metrics3, metrics2, metrics1, metrics0):                
  weightedF1 = (metrics0[5] *20 + metrics1[5] *20 + metrics2[5] *20 + metrics3[5] *20 + metrics5[5] *20 + metrics4[5] *20 + metrics6[5] *20 + metrics7[5] *20 + metrics8[5] *20 + metrics9[5]*20 + metrics10[5] *20 + metrics11[5] *20 + metrics12[5] *20 + metrics13[5] *20 + metrics14[5] *20 + metrics15[5] *20 + metrics16[5] *20 + metrics17[5] *20 + metrics18[5] *20 + metrics19[5] *20 + metrics20[5] *20 + metrics21[5] *20 + metrics22[5] *20 + metrics23[5] *20 + metrics24[5] *20 + metrics25[5] *20 + metrics26[5] *20 + metrics27[5] *20 + metrics28[5] *20 + metrics29[5] *20 + metrics30[5] *20 + metrics31[5] *20 + metrics32[5] *20 + metrics33[5] *20 + metrics34[5] *20 + metrics35[5] *20 + metrics36[5] *20 + metrics37[5] *20 + metrics38[5] *20 + metrics39[5] *20 + metrics40[5] *20 + metrics41[5] *20 + metrics42[5]*20)/(860)
  return(round(weightedF1,4))

#gewichteter F1 Score nach der Wichtigkeit der Schilder, wichtige Schilder werden doppelt gewertet
def weightedF1score2(metrics42,metrics41,metrics40,metrics39, metrics38, metrics37, metrics36, metrics35, metrics34, metrics33, metrics32, metrics31, metrics30, metrics29, metrics28, metrics27, metrics26, metrics25, metrics24, metrics23, metrics22, metrics21, metrics20, metrics19, metrics18, metrics17, metrics16, metrics15, metrics14, metrics13, metrics12, metrics11, metrics10, metrics9, metrics8, metrics7, metrics6, metrics5, metrics4, metrics3, metrics2, metrics1, metrics0):                
  weightedF1 = (metrics0[5] + metrics1[5] + metrics2[5] + metrics3[5] + metrics5[5] + metrics4[5] + metrics6[5] + metrics7[5] + metrics8[5] + metrics9[5] + metrics10[5] + metrics11[5] *3 + metrics12[5] *3 + metrics13[5] *3 + metrics14[5] *3 + metrics15[5] *3 + metrics16[5] + metrics17[5] *6 + metrics18[5] + metrics19[5] + metrics20[5] + metrics21[5] + metrics22[5] + metrics23[5] + metrics24[5] + metrics25[5] + metrics26[5] + metrics27[5] + metrics28[5] + metrics29[5] + metrics30[5] + metrics31[5] + metrics32[5] + metrics33[5] + metrics34[5] + metrics35[5] + metrics36[5] + metrics37[5] + metrics38[5] + metrics39[5] + metrics40[5] + metrics41[5] + metrics42[5])/(55)
  return(round(weightedF1,4))

#wichtige Schilder: Klasse 11 VOfahrt diese Kreuzung, Klasse 12 generelle Vorfahrt, klasse 13 Vorfahrt gewähren, Klasse 14 Stop, Klasse 15 einfahrverbot, Klasse 17 Einbahnstraße andere Seite


print('Accuracy: ', accuracy(predictions, test_labels))
print('----------------------------------------------------------------------------------------------------------------------')
print('TP = TruePositive, FP = FalsePositive, FN = FalseNegative')
print('Berechnung der Werte für jede Klasse und Zusammenfassung in mehreren globalen Metriken')
print('TP & FP & FN & Precision & Recall & F1_score Class0: ', MetricsClass0(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class1: ', MetricsClass1(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class2: ', MetricsClass2(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class3: ', MetricsClass3(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class4: ', MetricsClass4(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class5: ', MetricsClass5(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class6: ', MetricsClass6(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class7: ', MetricsClass7(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class8: ', MetricsClass8(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class9: ', MetricsClass9(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class10: ', MetricsClass10(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class11: ', MetricsClass11(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class12: ', MetricsClass12(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class13: ', MetricsClass13(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class14: ', MetricsClass14(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class15: ', MetricsClass15(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class16: ', MetricsClass16(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class17: ', MetricsClass17(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class18: ', MetricsClass18(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class19: ', MetricsClass19(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class20: ', MetricsClass20(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class21: ', MetricsClass21(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class22: ', MetricsClass22(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class23: ', MetricsClass23(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class24: ', MetricsClass24(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class25: ', MetricsClass25(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class26: ', MetricsClass26(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class27: ', MetricsClass27(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class28: ', MetricsClass28(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class29: ', MetricsClass29(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class30: ', MetricsClass30(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class31: ', MetricsClass31(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class32: ', MetricsClass32(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class33: ', MetricsClass33(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class34: ', MetricsClass34(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class35: ', MetricsClass35(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class36: ', MetricsClass36(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class37: ', MetricsClass37(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class38: ', MetricsClass38(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class39: ', MetricsClass39(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class40: ', MetricsClass40(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class41: ', MetricsClass41(predictions, test_labels))
print('TP & FP & FN & Precision & Recall & F1_score Class42: ', MetricsClass42(predictions, test_labels))
print('----------------------------------------------------------------------------------------------------------------------')
print('Micro F1-Score is calulated by using the total TP, total FP and total FN, same value as Accuracy is expected.')
print('Micro F1-Score:', microF1(metrics42, metrics41, metrics40, metrics39, metrics38, metrics37, metrics36, metrics35, metrics34, metrics33, metrics32, metrics31, metrics30, metrics29, metrics28, metrics27, metrics26, metrics25, metrics24, metrics23, metrics22, metrics21, metrics20, metrics19, metrics18, metrics17, metrics16, metrics15, metrics14, metrics13, metrics12, metrics11, metrics10, metrics9, metrics8, metrics7, metrics6, metrics5, metrics4, metrics3, metrics2, metrics1, metrics0))
print('----------------------------------------------------------------------------------------------------------------------')
print('Macro F1-Score is the unweighted mean of the F1-Score of every class')
print('Macro F1-Score:', macroF1(metrics42, metrics41, metrics40, metrics39, metrics38, metrics37, metrics36, metrics35, metrics34, metrics33, metrics32, metrics31, metrics30, metrics29, metrics28, metrics27, metrics26, metrics25, metrics24, metrics23, metrics22, metrics21, metrics20, metrics19, metrics18, metrics17, metrics16, metrics15, metrics14, metrics13, metrics12, metrics11, metrics10, metrics9, metrics8, metrics7, metrics6, metrics5, metrics4, metrics3, metrics2, metrics1, metrics0))
print('----------------------------------------------------------------------------------------------------------------------')
print('Sample amount is taken into consideration, weighted mean of the measures. Weights are the total number of samples for each class:')
print('Weighted F1-Score for sample amount:', weightedF1score(metrics42, metrics41, metrics40, metrics39, metrics38, metrics37, metrics36, metrics35, metrics34, metrics33, metrics32, metrics31, metrics30, metrics29, metrics28, metrics27, metrics26, metrics25, metrics24, metrics23, metrics22, metrics21, metrics20, metrics19, metrics18, metrics17, metrics16, metrics15, metrics14, metrics13, metrics12, metrics11, metrics10, metrics9, metrics8, metrics7, metrics6, metrics5, metrics4, metrics3, metrics2, metrics1, metrics0) )
print('----------------------------------------------------------------------------------------------------------------------')
print('Important signs are counted twice into the calculation:')
print('Weighted F1-Score for importance of signs:', weightedF1score2(metrics42, metrics41, metrics40, metrics39, metrics38, metrics37, metrics36, metrics35, metrics34, metrics33, metrics32, metrics31, metrics30, metrics29, metrics28, metrics27, metrics26, metrics25, metrics24, metrics23, metrics22, metrics21, metrics20, metrics19, metrics18, metrics17, metrics16, metrics15, metrics14, metrics13, metrics12, metrics11, metrics10, metrics9, metrics8, metrics7, metrics6, metrics5, metrics4, metrics3, metrics2, metrics1, metrics0))
print('----------------------------------------------------------------------------------------------------------------------')

# There is more and this should get you started: https://www.tensorflow.org/api_docs/python/tf/keras/metrics
# However it is not about how many metrics you crank out, it is about whether you find the meangingful ones and report on them.
# Think about a system on how to decide which metric to go for..

# You are looking for a great package to generate your reports, let me recommend https://plotly.com/dash/
