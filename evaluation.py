#task deepLerning TFE19-2, matricle number: 2374057

import numpy as np
import tensorflow as tf
import pandas as pd


class NetworkEvaluation:
    def __init__(self, predictions, classes):
        self.predictions = predictions
        self.classes = classes

    def printSingleClasses(self):
        true_predictions_per_class = np.zeros(len(self.classes), dtype=int)
        wrong_predictions_per_class = np.zeros(len(self.classes), dtype=int)
        total_per_class = np.zeros(len(self.classes), dtype=int)
        for i in range(0, len(self.classes)):
            if (np.array_equal(self.classes[i], self.predictions[i]) == True):
                true_predictions_per_class[(self.classes[i]-1)] = true_predictions_per_class[(self.classes[i]-1)]+1
                total_per_class[(self.classes[i]-1)] = total_per_class[(self.classes[i]-1)]+1

            else:
                wrong_predictions_per_class[(self.classes[i]-1)] = wrong_predictions_per_class[(self.classes[i]-1)]+1
                total_per_class[(self.classes[i]-1)] = total_per_class[(self.classes[i]-1)]+1

        for i in range(0, len(total_per_class)):
            if (total_per_class[i-1] != 0):
                with open("EvaluationReport.txt", "a") as txtfile:
                    print("Class {}:".format(i), "{} correct,".format(true_predictions_per_class[i-1]),
                          "{} wrong prediction(s)".format(wrong_predictions_per_class[i-1]),
                          "wrong with {} pictures of this class ->".format(total_per_class[i-1]),
                          "rel. acc. in %: {}".format(((true_predictions_per_class[i-1])/(total_per_class[i-1]))*100),
                          file=txtfile)

    def relativeAccuracy(self):
        metric = tf.keras.metrics.Accuracy()
        metric.update_state(self.predictions, self.classes)
        return metric.result().numpy()

    def printAbsoluteAccuracy(self):
        true_counter = 0
        false_counter = 0
        for i in range(0, len(self.classes)):
            if (np.array_equal(self.classes[i], self.predictions[i]) == True):
                true_counter = true_counter + 1
            else:
                false_counter = false_counter + 1
        with open("EvaluationReport.txt", "a") as txtfile:
            print("Absolute accuracy: {}".format(true_counter),
                  "/ {} pictures guessed correct ".format(true_counter+false_counter), file=txtfile)

    def printConfusionMatrix(self):
        classes = pd.Series(self.classes, name='Actual')
        predictions = pd.Series(self.predictions, name='Predicted')
        with open("EvaluationReport.txt", "a") as txtfile:
            print("\nConfusion Matrix for the test data:\n\n", pd.crosstab(classes, predictions), file=txtfile)

    def createEvaluationReport(self, model_name, batch_root):
        with open("EvaluationReport.txt", "w") as txtfile:
            print("Evaluation report for model {}:\n".format(model_name), file=txtfile)
            print("Data root for the used batch: {}\n".format(batch_root), file=txtfile)
            print("Accuracy of all classes combined:\n", file=txtfile)
        self.printAbsoluteAccuracy()
        acc = self.relativeAccuracy()
        with open("EvaluationReport.txt", "a") as txtfile:
            print("Relative accuracy: {} %\n".format(acc*100), file=txtfile)
            print("Data for all classes with pictures in the current test:\n", file=txtfile)
        self.printSingleClasses()
        self.printConfusionMatrix()
        with open("EvaluationReport.txt", "a") as txtfile:
            print("\nTest order:\n", file=txtfile)
        for i in range(1, len(self.classes)+1):
            with open("EvaluationReport.txt", "a") as txtfile:
                print("{}. Guess: ".format(i), self.predictions[i-1],
                      "predicted, {} is the actual class".format(self.classes[i-1]), file=txtfile)






