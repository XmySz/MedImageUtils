import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve


def find_optimal_cutoff(excel_path: str, sheet_name: str, true_label_col: str, pred_proba_col: str) -> float:
    """
    根据约登指数从Excel文件中计算最佳分类阈值。
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    y_true = df[true_label_col]
    y_pred_proba = df[pred_proba_col]

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    youden_index = tpr - fpr
    best_index = np.argmax(youden_index)
    optimal_cutoff = thresholds[best_index]

    return optimal_cutoff


def calculate_metrics_auc_acc(file_path, split_type=None, cutoff=0.5):
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    if split_type:
        data = df[df['split'] == split_type]
    else:
        data = df
    y_true = data['Label'].values
    y_prob = data['MRI_CRS'].values
    y_pred = (y_prob >= cutoff).astype(int)

    auc = roc_auc_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)

    n_bootstrap = 1000
    auc_scores = []
    acc_scores = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        y_pred_boot = (y_prob_boot >= cutoff).astype(int)
        auc_scores.append(roc_auc_score(y_true_boot, y_prob_boot))
        acc_scores.append(accuracy_score(y_true_boot, y_pred_boot))

    auc_ci = np.percentile(auc_scores, [2.5, 97.5])
    acc_ci = np.percentile(acc_scores, [2.5, 97.5])

    return f"{round(auc, 3)}[{round(auc_ci[0], 3)}-{round(auc_ci[1], 3)}]", f"{round(acc, 3)}[{round(acc_ci[0], 3)}-{round(acc_ci[1], 3)}]"


def calculate_metrics_sen_spe(file_path, split_type=None, cutoff=0.5):
    df = pd.read_excel(file_path, sheet_name='Sheet3')
    if split_type:
        data = df[df['split_fold_1'] == split_type]
    else:
        data = df
    y_true = data['Label'].values
    y_prob = data['Clinical_Predicted_Probability_fold1'].values
    y_pred = (y_prob >= cutoff).astype(int)

    tn = ((y_true == 0) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    n_bootstrap = 1000
    sen_scores = []
    spe_scores = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        y_pred_boot = (y_prob_boot >= cutoff).astype(int)

        tn_boot = ((y_true_boot == 0) & (y_pred_boot == 0)).sum()
        tp_boot = ((y_true_boot == 1) & (y_pred_boot == 1)).sum()
        fn_boot = ((y_true_boot == 1) & (y_pred_boot == 0)).sum()
        fp_boot = ((y_true_boot == 0) & (y_pred_boot == 1)).sum()

        sen_scores.append(tp_boot / (tp_boot + fn_boot))
        spe_scores.append(tn_boot / (tn_boot + fp_boot))

    sen_ci = np.percentile(sen_scores, [2.5, 97.5])
    spe_ci = np.percentile(spe_scores, [2.5, 97.5])

    return f"{round(sensitivity, 3)}[{round(sen_ci[0], 3)}-{round(sen_ci[1], 3)}]", f"{round(specificity, 3)}[{round(spe_ci[0], 3)}-{round(spe_ci[1], 3)}]"


def calculate_metrics_npv_ppv(file_path, split_type=None, cutoff=0.5):
    df = pd.read_excel(file_path, sheet_name='Sheet3')
    if split_type:
        data = df[df['split_fold_1'] == split_type]
    else:
        data = df
    y_true = data['Label'].values
    y_prob = data['Clinical_Predicted_Probability_fold1'].values
    y_pred = (y_prob >= cutoff).astype(int)

    tn = ((y_true == 0) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()

    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)

    n_bootstrap = 1000
    ppv_scores = []
    npv_scores = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        y_pred_boot = (y_prob_boot >= cutoff).astype(int)

        tn_boot = ((y_true_boot == 0) & (y_pred_boot == 0)).sum()
        tp_boot = ((y_true_boot == 1) & (y_pred_boot == 1)).sum()
        fn_boot = ((y_true_boot == 1) & (y_pred_boot == 0)).sum()
        fp_boot = ((y_true_boot == 0) & (y_pred_boot == 1)).sum()

        ppv_scores.append(tp_boot / (tp_boot + fp_boot))
        npv_scores.append(tn_boot / (tn_boot + fn_boot))

    ppv_ci = np.percentile(ppv_scores, [2.5, 97.5])
    npv_ci = np.percentile(npv_scores, [2.5, 97.5])

    return f"{round(ppv, 3)}[{round(ppv_ci[0], 3)}-{round(ppv_ci[1], 3)}]", f"{round(npv, 3)}[{round(npv_ci[0], 3)}-{round(npv_ci[1], 3)}]"


def calculate_metrics_f1(file_path, split_type=None, cutoff=0.5):
    df = pd.read_excel(file_path, sheet_name='Sheet3')
    if split_type:
        data = df[df['split_fold_1'] == split_type]
    else:
        data = df
    y_true = data['Label'].values
    y_prob = data['Clinical_Predicted_Probability_fold1'].values
    y_pred = (y_prob >= cutoff).astype(int)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()

    f1 = 2 * tp / (2 * tp + fp + fn)

    n_bootstrap = 200
    f1_scores = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        y_pred_boot = (y_prob_boot >= cutoff).astype(int)

        tp_boot = ((y_true_boot == 1) & (y_pred_boot == 1)).sum()
        fn_boot = ((y_true_boot == 1) & (y_pred_boot == 0)).sum()
        fp_boot = ((y_true_boot == 0) & (y_pred_boot == 1)).sum()

        f1_scores.append(2 * tp_boot / (2 * tp_boot + fp_boot + fn_boot))

    f1_ci = np.percentile(f1_scores, [2.5, 97.5])

    return f"{round(f1, 3)}[{round(f1_ci[0], 3)}-{round(f1_ci[1], 3)}]"


if __name__ == '__main__':
    print(calculate_metrics_auc_acc(r'D:\Data\OvarianCancer\Materials\NewLabel.xlsx', split_type='valid'))
    print(calculate_metrics_auc_acc(r'D:\Data\OvarianCancer\Materials\NewLabel.xlsx', split_type='test'))
    print(calculate_metrics_auc_acc(r'D:\Data\OvarianCancer\Materials\NewLabel.xlsx', split_type='test1'))
