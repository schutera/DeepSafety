import numpy as np
import itertools as it
import xlsxwriter
import translate_gt_and_predicts as trl


exportdict = {
    0: 20,
    1: 30,
    2: 50,
    3: 60,
    4: 70,
    5: 80,
    7: 100,
    8: 120,
}


class ConfusionMatrix:
    def __init__(self, combination):
        self.combination = combination

    tp, tn, fp, fn, f_a = 0, 0, 0, 0, 0


def create_cms(gt, pred):
    # Get all possible combinations for Confusion Matrices
    class_ids = np.unique(gt)
    combinations_forward = list(it.combinations(class_ids, 2))
    flipped_class_ids = np.flip(class_ids)
    combinations_backwards = list(it.combinations(flipped_class_ids, 2))
    all_combinations = combinations_forward + combinations_backwards
    all_cms_data = list()
    print('Beginning to create', int(np.size(all_combinations) / 2), 'Confusion Matrices...')

    # Get every single combination and create Confusion Matrices
    for i in range(int(np.size(all_combinations) / 2)):
        # Get a single combination single_combination(Class1, Class2)
        single_combination = all_combinations[i]
        cm_obj = ConfusionMatrix(single_combination)
        all_cms_data.append(cm_obj)

        # Filter for gt indices Class1
        gt_indices_class1 = [i for i, x in enumerate(gt) if x == single_combination[0]]

        # Filter for gt indices Class2
        gt_indices_class2 = [i for i, x in enumerate(gt) if x == single_combination[1]]

        # Create Confusion Matrix
        common_gt_indices = gt_indices_class1 + gt_indices_class2
        common_gt = np.zeros(np.size(common_gt_indices), dtype=int)
        common_pred = np.zeros(np.size(common_gt_indices), dtype=int)
        for j in range(np.size(common_gt_indices)):
            common_gt[j] = gt[common_gt_indices[j]]
        for j in range(np.size(common_gt_indices)):
            common_pred[j] = pred[common_gt_indices[j]]
        # TP = TruePos, TN = TrueNeg, FN = FalseNeg, TN = TrueNeg, f_a = FalselyAssigned
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


def calc_precision(all_cms_data):
    for i in range(len(all_cms_data)):
        all_cms_data[i].precision = all_cms_data[i].tp / (all_cms_data[i].tp + all_cms_data[i].fp)


def calc_recall(all_cms_data):
    for i in range(len(all_cms_data)):
        all_cms_data[i].recall = all_cms_data[i].tp / (all_cms_data[i].tp + all_cms_data[i].fn)


def calc_f1(all_cms_data):
    for i in range(len(all_cms_data)):
        all_cms_data[i].f1 = 2 * (all_cms_data[i].precision * all_cms_data[i].recall) /\
                             (all_cms_data[i].precision + all_cms_data[i].recall)


def export_data_excel(all_cms_data, gt):
    class_ids = np.unique(gt)
    workbook = xlsxwriter.Workbook('Evaluation_data_speed_signs.xlsx')
    worksheets = list()
    all_cms_data.sort(key=lambda x: x.combination, reverse=False)

    bold_f = workbook.add_format({'bold': True})
    tp_tn_f = workbook.add_format({'bg_color': '#319dc2', 'border': 1})
    fn_fp_f = workbook.add_format({'bg_color': '#86d3e3', 'border': 1})
    class_f = workbook.add_format({'bg_color': '#a69686', 'border': 1})
    f_a_f = workbook.add_format({'bg_color': '#e28743', 'border': 1})
    fv_r = workbook.add_format({'bg_color': '#E21C1C', 'border': 1})
    fv_g = workbook.add_format({'bg_color': '#4BD045', 'border': 1})
    border_f = workbook.add_format({'border': 1})

    for i in range(int(np.size(class_ids))):
        worksheets.append(workbook.add_worksheet('Class' + trl.transldict[class_ids[i]]))
        factor = i * (np.size(class_ids) - 1)
        k = 0

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
        worksheets[i].write(12, 0, 'Precision:', border_f)
        worksheets[i].write(12, 1, 'Recall:', border_f)
        worksheets[i].write(12, 2, 'F1-Score:', border_f)

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
            worksheets[i].write(2, k + 3, exportdict[all_cms_data[j + factor].combination[1]] /
                                exportdict[all_cms_data[j + factor].combination[0]], fv_r)
            worksheets[i].write(3, k + 3, all_cms_data[j + factor].f_a, f_a_f)
            worksheets[i].write(4, k + 0, 'Precision: ' + str(all_cms_data[j + factor].precision), border_f)
            worksheets[i].write(4, k + 1, 'Recall: ' + str(all_cms_data[j + factor].recall), border_f)
            worksheets[i].write(4, k + 2, 'F1-Score: ' + str(all_cms_data[j + factor].f1), border_f)
            worksheets[i].conditional_format(2, k + 3, 2, k + 3, {'type': 'cell',
                                                        'criteria': 'between',
                                                        'minimum': 0.75,
                                                        'maximum': 1,
                                                        'format': fv_g})
            k = k + 5
    workbook.close()
