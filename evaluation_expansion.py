import numpy as np
import itertools as it
import xlsxwriter
import translate_gt_and_predicts as trl

# This dict is necessary to calculate the 'FaultValue' later on
export_dict = {
    0: 20,
    1: 30,
    2: 50,
    3: 60,
    4: 70,
    5: 80,
    7: 100,
    8: 120,
}


# This class defines the necessary ConfusionMatrix data
class ConfusionMatrix:
    def __init__(self, combination):
        self.combination = combination

    tp, tn, fp, fn, f_a = 0, 0, 0, 0, 0


# This function creates the ConfusionMatrix datasets for all classes against each other
def create_cms(gt, pred):
    # Getting all possible combinations for Confusion Matrices
    class_ids = np.unique(gt)
    combinations_forward = list(it.combinations(class_ids, 2))
    flipped_class_ids = np.flip(class_ids)
    combinations_backwards = list(it.combinations(flipped_class_ids, 2))
    all_combinations = combinations_forward + combinations_backwards

    # Creating a list to store the CM objects
    all_cms_data = list()
    print('Beginning to create', int(np.size(all_combinations) / 2), 'Confusion Matrices...')

    # Creating Confusion Matrices for all combinations
    for i in range(int(np.size(all_combinations) / 2)):
        # Getting a single combination and create object CM
        single_combination = all_combinations[i]
        cm_obj = ConfusionMatrix(single_combination)
        all_cms_data.append(cm_obj)

        # Filtering for gt indices Class1
        gt_indices_class1 = [i for i, x in enumerate(gt) if x == single_combination[0]]

        # Filtering for gt indices Class2
        gt_indices_class2 = [i for i, x in enumerate(gt) if x == single_combination[1]]

        # Creates two arrays which will later contain the common gt and pred data
        common_gt_indices = gt_indices_class1 + gt_indices_class2
        common_gt = np.zeros(np.size(common_gt_indices), dtype=int)
        common_pred = np.zeros(np.size(common_gt_indices), dtype=int)

        # Getting the gt values at the filtered gt indices and the associated preds
        for j in range(np.size(common_gt_indices)):
            common_gt[j] = gt[common_gt_indices[j]]
        for j in range(np.size(common_gt_indices)):
            common_pred[j] = pred[common_gt_indices[j]]

        # Calculating CM necessary values TP, TN, FP, FN and stores them
        for j in range(np.size(common_gt)):
            if common_gt[j] == common_pred[j] == single_combination[0]:
                cm_obj.tp += 1
            elif common_gt[j] == common_pred[j] == single_combination[1]:
                cm_obj.tn += 1
            elif common_gt[j] == single_combination[1] and common_pred[j] == single_combination[0]:
                cm_obj.fp += 1
            elif common_gt[j] == single_combination[0] and common_pred[j] == single_combination[1]:
                cm_obj.fn += 1
            else:
                cm_obj.f_a += 1
        cm_obj.cm = np.array([[cm_obj.tp, cm_obj.fn], [cm_obj.fp, cm_obj.tn]])
    print(len(all_cms_data), 'Confusion Matrices were created')
    return all_cms_data


# This function calculates the precision score for all CM objects
def calc_precision(all_cms_data):
    for i in range(len(all_cms_data)):
        all_cms_data[i].precision = all_cms_data[i].tp / (all_cms_data[i].tp + all_cms_data[i].fp)


# This function calculates the recall score for all CM objects
def calc_recall(all_cms_data):
    for i in range(len(all_cms_data)):
        all_cms_data[i].recall = all_cms_data[i].tp / (all_cms_data[i].tp + all_cms_data[i].fn)


# This function calculates the f1 score for all CM objects
def calc_f1(all_cms_data):
    for i in range(len(all_cms_data)):
        all_cms_data[i].f1 = 2 * (all_cms_data[i].precision * all_cms_data[i].recall) / \
                             (all_cms_data[i].precision + all_cms_data[i].recall)


# This function exports all calculated data to an .xlsx file
def export_data_excel(all_cms_data, gt):
    # Creating a workbook object for .xlsx file
    class_ids = np.unique(gt)
    workbook = xlsxwriter.Workbook('Evaluation_data_speed_signs.xlsx')
    worksheets = list()
    all_cms_data.sort(key=lambda x: x.combination, reverse=False)

    # Defining some formats for the export
    bold_f = workbook.add_format({'bold': True})
    tp_tn_f = workbook.add_format({'bg_color': '#319dc2', 'border': 1})
    fn_fp_f = workbook.add_format({'bg_color': '#86d3e3', 'border': 1})
    class_f = workbook.add_format({'bg_color': '#a69686', 'border': 1})
    f_a_f = workbook.add_format({'bg_color': '#e28743', 'border': 1})
    fv_r = workbook.add_format({'bg_color': '#E21C1C', 'border': 1})
    fv_g = workbook.add_format({'bg_color': '#4BD045', 'border': 1})
    scores_f = workbook.add_format({'border': 1})

    # Writing the data to .xlsx file
    for i in range(int(np.size(class_ids))):
        # Creating a worksheet for every class
        worksheets.append(workbook.add_worksheet('Class' + trl.transldict[class_ids[i]]))
        factor = i * (np.size(class_ids) - 1)
        k = 0

        # Creating a legend in the .xlsx file
        worksheets[i].write(8, 0, 'Combination: Class1, Class2', bold_f)
        worksheets[i].write(9, 0, 'Class1', class_f)
        worksheets[i].write(10, 0, 'Class2', class_f)
        worksheets[i].write(8, 1, 'Class1', class_f)
        worksheets[i].write(8, 2, 'Class2', class_f)
        worksheets[i].write(9, 1, 'TP', tp_tn_f)
        worksheets[i].write(9, 2, 'FN', fn_fp_f)
        worksheets[i].write(10, 1, 'FP', fn_fp_f)
        worksheets[i].write(10, 2, 'TN', tp_tn_f)
        worksheets[i].write(10, 3, 'CriticalValue', fv_g)
        worksheets[i].write(11, 3, 'FalselyAssociated', f_a_f)
        worksheets[i].write(12, 0, 'Precision:', scores_f)
        worksheets[i].write(12, 1, 'Recall:', scores_f)
        worksheets[i].write(12, 2, 'F1-Score:', scores_f)

        # Writing the CMs for each class in its worksheet
        for j in range(int(np.size(class_ids)) - 1):
            combination_as_str = 'Class Combination: ' + str(all_cms_data[j + factor].combination[0]) + '\\' + \
                                 str(all_cms_data[j + factor].combination[1])
            worksheets[i].write(0, k + 0, combination_as_str, bold_f)
            worksheets[i].set_column(0, k + 5, len('Combination: Class1 \\ Class2'))
            worksheets[i].write(1, k + 0, trl.transldict[all_cms_data[j + factor].combination[0]], class_f)
            worksheets[i].write(2, k + 0, trl.transldict[all_cms_data[j + factor].combination[1]], class_f)
            worksheets[i].write(0, k + 1, trl.transldict[all_cms_data[j + factor].combination[0]], class_f)
            worksheets[i].write(0, k + 2, trl.transldict[all_cms_data[j + factor].combination[1]], class_f)
            worksheets[i].write(1, k + 1, all_cms_data[j + factor].tp, tp_tn_f)
            worksheets[i].write(1, k + 2, all_cms_data[j + factor].fn, fn_fp_f)
            worksheets[i].write(2, k + 1, all_cms_data[j + factor].fp, fn_fp_f)
            worksheets[i].write(2, k + 2, all_cms_data[j + factor].tn, tp_tn_f)
            worksheets[i].write(2, k + 3, export_dict[all_cms_data[j + factor].combination[1]] /
                                export_dict[all_cms_data[j + factor].combination[0]], fv_r)
            worksheets[i].write(3, k + 3, all_cms_data[j + factor].f_a, f_a_f)
            worksheets[i].write(4, k + 0, 'Precision: ' + str(all_cms_data[j + factor].precision), scores_f)
            worksheets[i].write(4, k + 1, 'Recall: ' + str(all_cms_data[j + factor].recall), scores_f)
            worksheets[i].write(4, k + 2, 'F1-Score: ' + str(all_cms_data[j + factor].f1), scores_f)
            worksheets[i].conditional_format(2, k + 3, 2, k + 3, {'type': 'cell',
                                                                  'criteria': 'between',
                                                                  'minimum': 0.75,
                                                                  'maximum': 1,
                                                                  'format': fv_g})
            k = k + 5
    workbook.close()
