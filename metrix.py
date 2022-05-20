import numpy as np
import pandas as pd
from enum import Enum


class Metric(Enum):
    """Enumeration to select different metrics."""

    ACCURACY = 1
    CONFIDENCE = 2
    HIGH_CONFIDENCE = 3
    FALSE_CONFIDENCE = 4
    MISSING_CONFIDENCE = 5


class Metrics:
    """A Metrics object can return certain metrics on demand."""

    def __get_predictions(model_output):
        """Get the predictions from a model's output layer.

        The prediction is the highest value of the output.

        :param model_output: the model's output
        :type model_output: 2D array
        :return: the predictions
        :rtype: 1D array
        """

        return np.argmax(model_output, axis=1)

    def __get_confidences(model_output):
        """Get the confidences from a model's output.

        The confidence is calculated as the relative distance from
        the highest value to the second highest value.
        For this task, the model's output is normalized so that the
        minimum value becomes 0 and the maximum value becomes 1
        according to this formula:

        normalized = (value - min(values)) / (max(values) - min(values))

        The confidence score is 1 - the second highest normalized value.

        :param model_output: the model's output
        :type model_output: 2D array
        :return: the confidences
        :rtype: 1D array
        """

        # shift and scale the interval so min lies at 0 and max lies at 1
        shape = np.shape(model_output)
        repetitions = shape[1]
        min = np.repeat(np.min(model_output, axis=1), repetitions, axis=0).reshape(
            shape
        )
        max = np.repeat(np.max(model_output, axis=1), repetitions, axis=0).reshape(
            shape
        )
        diff = max - min

        normalized = (
            model_output - min
        ) / diff  # division by 0 results in inf, no crash

        shape = normalized.shape
        if len(shape) != 2:
            print("Wrong shape! Should be 2D array!")
            return 0
        if shape[1] < 2:
            print("Accuracy is not defined for models with less than 2 output classes!")
            return np.zeros(shape[1])

        # sort the results to get 2nd highest value
        return 1 - np.sort(normalized, axis=1)[:, -2]

    def __init__(self, model_output, ground_truth):
        """Create a Metrics object.

        A Metrics object stores the ground truth, predictions and
        confidences in a dataframe. All statistical calculations
        are then performed using the dataframe, caching certain
        intermediate results as additional columns.

        :param model_output: the model's output
        :type model_output: 2D array
        :param ground_truth: the ground truth
        :type ground_truth: 1D array
        """

        self.__df = pd.DataFrame(
            {
                self.__ground_truth_column(): ground_truth,
                self.__prediction_column(): Metrics.__get_predictions(model_output),
                self.__confidence_column(): Metrics.__get_confidences(model_output),
            }
        )

    def __ground_truth_column(self):
        """Get the name of the ground truth column.

        :return: name of the ground truth column
        :rtype: str
        """

        return "ground truth"

    def __prediction_column(self):
        """Get the name of the prediction column.

        :return: name of the prediction column
        :rtype: str
        """

        return "prediction"

    def __confidence_column(self):
        """Get the name of the confidence column.

        :return: name of the confidence column
        :rtype: str
        """

        return "confidence"

    def __accuracy_column(self):
        """Get the name of the accuracy column.

        If the column doesn't exists, create it and thereby
        cache the calculated accuracies.

        :return: name of the accuracy column
        :rtype: str
        """

        accurate_column = "accuracy"
        ground_truth_column = self.__ground_truth_column()
        prediction_column = self.__prediction_column()

        if accurate_column not in self.__df.columns:
            self.__df[accurate_column] = (
                self.__df[ground_truth_column] == self.__df[prediction_column]
            )

        return accurate_column

    def __high_confidence_column(self, thres):
        """Get the name of the high confidence column with the
        given threshold.

        If the column doesn't exists, create it and thereby
        cache the calculated high confidences.

        A confidence is considered high when it is greater than
        the given threshold value.

        :param thres: threshold value to treat confidence as high confidence
        :type thres: float
        :return: name of the high confidence column with the given threshold
        :rtype: str
        """

        high_confidence_column = f"high confidence (threshold = {thres})"
        confidence_column = self.__confidence_column()

        if high_confidence_column not in self.__df.columns:
            self.__df[high_confidence_column] = self.__df[confidence_column] > thres

        return high_confidence_column

    def __false_confidence_column(self, thres):
        """Get the name of the false confidence column with the
        given threshold.

        If the column doesn't exists, create it and thereby
        cache the calculated false confidences.

        False confidence is when the model's prediction is incorrect
        but the confidence is considered high.

        :param thres: threshold value to treat confidence as high confidence
        :type thres: float
        :return: name of the false confidence column with the given threshold
        :rtype: str
        """

        false_confidence_column = f"false confidence (threshold = {thres})"
        high_confidence_column = self.__high_confidence_column(thres)
        accurate_column = self.__accuracy_column()

        if false_confidence_column not in self.__df.columns:
            self.__df[false_confidence_column] = (
                self.__df[high_confidence_column] & ~self.__df[accurate_column]
            )

        return false_confidence_column

    def __missing_confidence_column(self, thres):
        """Get the name of the missing confidence column with the
        given threshold.

        If the column doesn't exists, create it and thereby
        cache the calculated missing confidences.

        Missing confidence is when the model's prediction is correct
        but the confidence is not considered high.

        :param thres: threshold value to treat confidence as high confidence
        :type thres: float
        :return: name of the missing confidence column with the given threshold
        :rtype: str
        """

        missing_confidence_column = f"missing confidence (threshold = {thres})"
        high_confidence_column = self.__high_confidence_column(thres)
        accurate_column = self.__accuracy_column()

        if missing_confidence_column not in self.__df.columns:
            self.__df[missing_confidence_column] = (
                ~self.__df[high_confidence_column] & self.__df[accurate_column]
            )

        return missing_confidence_column

    def __metric_to_column(self, metric, hc_thres, fc_thres, mc_thres):
        """Convert a metric from the Metric enum to the corresponding
        column name with the respective threshold.

        :param metric: the metric from the Metric enum
        :type metric: Metric
        :param hc_thres: the threshold used for high confidence
        :type hc_thres: float
        :param fc_thres: the threshold used for false confidence
        :type fc_thres: float
        :param mc_thres: the threshold used for missing confidence
        :type mc_thres: float
        :return: the corresponding column name
        :rtype: str
        """

        if metric is Metric.ACCURACY:
            return self.__accuracy_column()
        elif metric is Metric.CONFIDENCE:
            return self.__confidence_column()
        elif metric is Metric.HIGH_CONFIDENCE:
            return self.__high_confidence_column(hc_thres)
        elif metric is Metric.FALSE_CONFIDENCE:
            return self.__false_confidence_column(fc_thres)
        elif metric is Metric.MISSING_CONFIDENCE:
            return self.__missing_confidence_column(mc_thres)

    def overall_mean(
        self, *metrics, thres=0.25, hc_thres=None, fc_thres=None, mc_thres=None
    ) -> pd.Series:
        """Compute the overall mean values for the given metrics.

        :param thres: the threshold value used everywhere, defaults to 0.25
        :type thres: float, optional
        :param hc_thres: the threshold value used for high confidence, defaults to None
        :type hc_thres: float, optional
        :param fc_thres: the threshold value used for false confidence, defaults to None
        :type fc_thres: float, optional
        :param mc_thres: the threshold value used for missing confidence, defaults to None
        :type mc_thres: float, optional
        :return: the resulting mean values
        :rtype: pd.Series
        """

        columns = [
            self.__metric_to_column(
                metric,
                thres if hc_thres is None else hc_thres,
                thres if fc_thres is None else fc_thres,
                thres if mc_thres is None else mc_thres,
            )
            for metric in metrics
        ]

        return self.__df[columns].mean()

    def per_class_mean(
        self, *metrics, thres=0.25, hc_thres=None, fc_thres=None, mc_thres=None
    ) -> pd.DataFrame:
        """Compute the mean values for the given metrics per class.

        :param thres: the threshold value used everywhere, defaults to 0.25
        :type thres: float, optional
        :param hc_thres: the threshold value used for high confidence, defaults to None
        :type hc_thres: float, optional
        :param fc_thres: the threshold value used for false confidence, defaults to None
        :type fc_thres: float, optional
        :param mc_thres: the threshold value used for missing confidence, defaults to None
        :type mc_thres: float, optional
        :return: the resulting mean values per class
        :rtype: pd.DataFrame
        """

        columns = [
            self.__metric_to_column(
                metric,
                thres if hc_thres is None else hc_thres,
                thres if fc_thres is None else fc_thres,
                thres if mc_thres is None else mc_thres,
            )
            for metric in metrics
        ]
        ground_truth_column = self.__ground_truth_column()
        columns.append(ground_truth_column)

        return self.__df[columns].groupby(ground_truth_column).mean()

    def plot_per_class_mean(
        self,
        *metrics,
        thres=0.25,
        hc_thres=None,
        fc_thres=None,
        mc_thres=None,
        file_name=None,
    ):
        """Plot the mean values for the given metrics per class.

        Save it in file_name or show it if not provided.

        :param thres: the threshold value used everywhere, defaults to 0.25
        :type thres: float, optional
        :param hc_thres: the threshold value used for high confidence, defaults to None
        :type hc_thres: float, optional
        :param fc_thres: the threshold value used for false confidence, defaults to None
        :type fc_thres: float, optional
        :param mc_thres: the threshold value used for missing confidence, defaults to None
        :type mc_thres: float, optional
        :param file_name: the file name to save the plot to, defaults to None
        :type file_name: str, optional
        """

        ax = self.per_class_mean(
            *metrics,
            thres=thres,
            hc_thres=hc_thres,
            fc_thres=fc_thres,
            mc_thres=mc_thres,
        ).plot.bar()
        fig = ax.get_figure()

        if file_name is None:
            fig.show()
        else:
            fig.savefig(file_name)

    def print_report(self, safe_accuracy=0.95):
        """Print a report about the most important metrics.

        :param safe_accuracy: the accuracy the model has to achive to be considered safe, defaults to 0.95
        :type safe_accuracy: float, optional
        """

        print("")

        accuracy = self.overall_mean(Metric.ACCURACY)[0]
        print(
            "The model achived an overall accuracy of {:.2f}%.".format(accuracy * 100)
        )

        if accuracy < safe_accuracy:
            print("This is not enough to be considered safe!")

        print("")

        # check accuracies for each class
        accuracies = self.per_class_mean(Metric.ACCURACY)
        count = 0
        for index, row in accuracies.iterrows():
            accuracy = row[0]

            if accuracy < safe_accuracy:
                count += 1
                print(
                    "Class {} doesn't have have enough accuracy with {:.2f}%.".format(
                        index, accuracy * 100
                    )
                )
        if count > 0:
            print(
                "This is a hint that the training data might be incomplete for {}.".format(
                    "this class" if count == 1 else "these classes"
                )
            )
            print("Please review your respective training data in terms of variance!")

        print("")

        # check false confidence for each class
        fcs = self.per_class_mean(Metric.FALSE_CONFIDENCE)
        min = fcs.min()[0]
        median = fcs.median()[0]
        count = 0
        for index, row in fcs.iterrows():
            fc = row[0]
            # compare the current false confidence (fc) to the median
            # info if current fc lies further apart than minimum fc
            # in absolute distance and current fc must also be greater
            if fc - median > median - min:
                count += 1
                print(
                    "Class {} has pretty high false confidence with {:.2f}%.".format(
                        index, fc * 100
                    )
                )
        if count > 0:
            print(
                "{} false confidence deviates more from the median then the one with the least false confidence.".format(
                    "This class's" if count == 1 else "These classes'"
                )
            )
            print(
                "Check if {} very similar to other classes!".format(
                    "this class is" if count == 1 else "these classes are"
                )
            )

        print("")
