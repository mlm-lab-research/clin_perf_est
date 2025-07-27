import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
from utils.plots import *
import sklearn
from torch.utils.data import DataLoader, Subset
from utils.metrics import calculate_metrics


def CBPE_accuracy(outs, balanced=False, prevalence_correction=False):
    """
    Calculate the estimated accuracy of the model outputs.

    Parameters:
    outs (list or numpy array): List or array of model output scores.

    Returns:
    float: Estimated accuracy.
    """    
    # Calculate estimated performance metrics
    TP = []
    TN = []
    FP = []
    FN = []

    for score in outs:
        pred = np.round(score)
        p_not_eq = np.abs(pred - score)
        p_eq = 1 - p_not_eq 
    
        if pred == 1:
            TP.append(p_eq)
            FP.append(p_not_eq)
        else:
            TN.append(p_eq)
            FN.append(p_not_eq)

    if not balanced:
        # prevalence corrected accuracy
        if prevalence_correction:
            est_prev = estimate_prior_em(outs)
            acc_estim = est_prev * (np.sum(TP) / (np.sum(TP) + np.sum(FN))) + (1 - est_prev) * (np.sum(TN) / (np.sum(TN) + np.sum(FP)))
        else:
            acc_estim = (np.sum(TP) + np.sum(TN)) / len(outs)
    else:
        acc_estim = 0.5 * (np.sum(TP)/(np.sum(TP) + np.sum(FN)) + np.sum(TN)/(np.sum(TN) + np.sum(FP)))
    return acc_estim

def CBPE_auroc(outs, comet_logger=None, cfg=None, class_names=None, show_plots=True):
    """
    Calculate the estimated AUROC (Area Under the Receiver Operating Characteristic curve) of the model outputs.

    Parameters:
    outs (list or numpy array): List or array of model output scores.
    comet_logger (object): Logger object for logging the ROC curve.
    cfg (object): Configuration object.
    model_pathology (str): pathology name.

    Returns:
    None
    """
    # Calculate estimated performance metrics
    thresholds_ = np.quantile(outs, q=np.linspace(0, 1, 100))
    thresholds = np.unique(thresholds_)

    for t in thresholds:
        TP = []
        TN = []
        FP = []
        FN = []

        for score in outs:
            # Round based on threshold t
            pred = (score >= t).astype(int)
            
            p_not_eq = np.abs(pred - score)
            p_eq = 1 - p_not_eq 
        
            if pred == 1:
                TP.append(p_eq)
                FP.append(p_not_eq)
            else:
                TN.append(p_eq)
                FN.append(p_not_eq)

        TPR_list.append(np.sum(TP) / (np.sum(TP) + np.sum(FN)))
        FPR_list.append(np.sum(FP) / (np.sum(FP) + np.sum(TN)))
    
    fig = plt.figure()
    plt.plot(FPR_list, TPR_list)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC curve {class_names}')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    if comet_logger is not None and cfg is not None:
        if cfg.comet_logger.initialize:
            comet_log_figure(comet_logger, fig, name=f'ROC curve {class_names}', step=0, cfg=cfg)
    if show_plots:
        plt.show()
    plt.close(fig)

    auroc_estim = np.trapz(TPR_list[::-1], FPR_list[::-1])
    return auroc_estim

def CBPE_F1(outs, prevalence_correction=False):
    """
    Calculate the estimated F1 score of the model outputs.

    Parameters:
    outs (list or numpy array): List or array of model output scores.

    Returns:
    float: Estimated F1 score.
    """
    # Calculate estimated performance metrics
    TP = []
    FP = []
    FN = []
    TN = []

    for score in outs:
        pred = np.round(score)
        p_not_eq = np.abs(pred - score)
        p_eq = 1 - p_not_eq 
    
        if pred == 1:
            TP.append(p_eq)
            FP.append(p_not_eq)
        else:
            FN.append(p_not_eq)
            TN.append(p_eq)
    

    tp = np.sum(TP)
    fp = np.sum(FP)
    fn = np.sum(FN)
    tn = np.sum(TN)

    # Recall stays unchanged (intrinsic to the classifier)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # False Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Prevalence-corrected precision (i.e., estimated PPV)
    if prevalence_correction:
        prev_est = estimate_prior_em(outs)
        precision = (
            prev_est * recall /
            (prev_est * recall + (1 - prev_est) * fpr)
        ) if (prev_est * recall + (1 - prev_est) * fpr) > 0 else 0.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Now compute corrected F1
    if precision + recall == 0:
        f1_estim = 0.0
    else:
        f1_estim = 2 * (precision * recall) / (precision + recall)

    return f1_estim

def CBPE_plr(outs):
    # Calculate estimated performance metrics
    TP = []
    FP = []
    FN = []
    TN = []

    for score in outs:
        pred = np.round(score)
        p_not_eq = np.abs(pred - score)
        p_eq = 1 - p_not_eq 
    
        if pred == 1:
            TP.append(p_eq)
            FP.append(p_not_eq)
        else:
            FN.append(p_not_eq)
            TN.append(p_eq)
    
    recall = np.sum(TP) / (np.sum(TP) + np.sum(FN))
    specificity = np.sum(TN) / (np.sum(TN) + np.sum(FP))
    plr = recall / (1 - specificity)
    return plr

def CBPE_confusion_matrix(outs):
    """
    Calculate the estimated confusion matrix of the model outputs.

    Parameters:
    outs (list or numpy array): TP, FP, FN, TN.

    Returns:
    numpy array: Estimated confusion matrix.
    """
    # Calculate estimated performance metrics
    TP = []
    FP = []
    FN = []
    TN = []

    for score in outs:
        pred = np.round(score)
        p_not_eq = np.abs(pred - score)
        p_eq = 1 - p_not_eq 
    
        if pred == 1:
            TP.append(p_eq)
            FP.append(p_not_eq)
        else:
            FN.append(p_not_eq)
            TN.append(p_eq)
    TP = np.sum(TP)
    FP = np.sum(FP)
    FN = np.sum(FN)
    TN = np.sum(TN)


    return TP, FP, TN, FN

def CBPE_precision(outs, prevalence_correction=False):
    TP, FP, TN, FN = CBPE_confusion_matrix(outs)
    if prevalence_correction:
        est_prev = estimate_prior_em(outs)
        precision = est_prev * TP / (est_prev * TP + (1-est_prev)*FP) if (est_prev * TP + (1-est_prev) * FP) > 0 else 0.0
    else:
        denominator = (TP + FP)
        if denominator == 0:
            return 0.0
        precision = TP / denominator
    return precision

def CBPE_recall(outs):
    """
    Calculate the recall based on the CBPE confusion matrix.
    Recall = TP / (TP + FN)
    """
    TP, FP, TN, FN = CBPE_confusion_matrix(outs)
    denominator = (TP + FN)
    if denominator == 0:
        return 0.0
    return TP / denominator

def CBPE_specificity(outs):
    """
    Calculate the specificity based on the CBPE confusion matrix.
    Specificity = TN / (TN + FP)
    """
    TP, FP, TN, FN = CBPE_confusion_matrix(outs)
    denominator = (TN + FP)
    if denominator == 0:
        return 0.0
    return TN / denominator


# MICCAI PAPER METHODS

# DoC
def CM_DoC_metric_estim(val_outs, val_labels, test_outs, prevalence_correction=False):
    val_arr   = np.asarray(val_outs,  dtype=float).ravel()
    test_arr  = np.asarray(test_outs, dtype=float).ravel()
    labels    = np.asarray(val_labels, dtype=int).ravel()
    n_test    = test_arr.size

    ts_ = np.linspace(0.001, 0.999, 100)
    ts = np.quantile(test_outs, q=ts_)
    ts = np.unique(ts)

    mask_val_pos  = val_arr[None, :]  >= ts[:, None]
    mask_test_pos = test_arr[None, :] >= ts[:, None]

    count_val_pos  = mask_val_pos.sum(axis=1)
    count_test_pos = mask_test_pos.sum(axis=1)
    count_val_neg  = (~mask_val_pos).sum(axis=1)
    count_test_neg = (~mask_test_pos).sum(axis=1)

    tp_val = ((labels[None, :] == 1) & mask_val_pos).sum(axis=1)
    fp_val = ((labels[None, :] == 0) & mask_val_pos).sum(axis=1)
    fn_val = ((labels[None, :] == 1) & ~mask_val_pos).sum(axis=1)
    tn_val = ((labels[None, :] == 0) & ~mask_val_pos).sum(axis=1)


    val_precision = np.divide(
        tp_val, tp_val + fp_val,
        out=np.zeros_like(tp_val, dtype=float),
        where=(tp_val + fp_val) > 0
    )
    
    val_npv = np.divide(
        tn_val, tn_val + fn_val,
        out=np.zeros_like(tn_val, dtype=float),
        where=(tn_val + fn_val) > 0
    )
    
    # sum s_i , s_i >= 0.5
    sum_val_pos  = (mask_val_pos  * val_arr[None, :]).sum(axis=1)
    sum_val_neg  = ((~mask_val_pos) * (1-val_arr[None, :])).sum(axis=1)
    
    # print(sum_val_neg)
    sum_test_pos = (mask_test_pos * test_arr[None, :]).sum(axis=1)
    sum_test_neg = ((~mask_test_pos) * (1-test_arr[None, :])).sum(axis=1)

    mean_val1  = np.divide(
        sum_val_pos, count_val_pos,
        out=np.zeros_like(sum_val_pos),
        where=count_val_pos > 0
    )
    mean_val0  = np.divide(
        sum_val_neg, count_val_neg,
        out=np.zeros_like(sum_val_neg),
        where=count_val_neg > 0
    )
    
    mean_test1 = np.divide(
        sum_test_pos, count_test_pos,
        out=np.zeros_like(sum_test_pos),
        where=count_test_pos > 0
    )
    mean_test0 = np.divide(
        sum_test_neg, count_test_neg,
        out=np.zeros_like(sum_test_neg),
        where=count_test_neg > 0
    )

    ood_precision = np.clip(val_precision + mean_test1 - mean_val1, 0, 1)
    ood_npv       = np.clip(val_npv       + mean_test0 - mean_val0, 0, 1)

    est_TP = ood_precision * count_test_pos
    est_FP = count_test_pos - est_TP
    est_TN = ood_npv      * count_test_neg
    est_FN = count_test_neg - est_TN


    tpr = np.divide(
        est_TP, est_TP + est_FN,
        out=np.zeros_like(est_TP),
        where=(est_TP + est_FN) > 0
    )
    fpr = np.divide(
        est_FP, est_FP + est_TN,
        out=np.zeros_like(est_FP),
        where=(est_FP + est_TN) > 0
    )

    auc = np.trapz(tpr[::-1], fpr[::-1])

    idx_mid = np.abs(ts - 0.5).argmin()
    tp_m, fp_m = est_TP[idx_mid], est_FP[idx_mid]
    tn_m, fn_m = est_TN[idx_mid], est_FN[idx_mid]

    # Prevalence correct precision, f1 and acc
    tpr = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0.0
    fpr = fp_m / (fp_m + tn_m) if (fp_m + tn_m) > 0 else 0.0
    tnr = tn_m / (tn_m + fp_m) if (tn_m + fp_m) > 0 else 0.0
    if prevalence_correction:
        prev_est = estimate_prior_em(test_outs)
        prec_m = (prev_est * tpr) / (prev_est * tpr + (1 - prev_est) * fpr) if (prev_est * tpr + (1 - prev_est) * fpr) > 0 else 0.0
        accuracy = prev_est * tpr + (1 - prev_est) * tnr
    else:
        prec_m = tp_m / (tp_m + fp_m) if (tp_m + fp_m) > 0 else 0.0
        accuracy = (tp_m + tn_m) / (tp_m + tn_m + fp_m + fn_m) if (tp_m + tn_m + fp_m + fn_m) > 0 else 0.0

       
    bal_accuracy = 0.5 * (
        (tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0.0)
      + (tn_m / (tn_m + fp_m) if (tn_m + fp_m) > 0 else 0.0))
    rec_m = tpr
    specificity = tn_m / (tn_m + fp_m) if (tn_m + fp_m) > 0 else 0.0
    
    f1_score    = 2 * prec_m * rec_m / (prec_m + rec_m) if (prec_m + rec_m) > 0 else 0.0

    return {
        'accuracy':     accuracy,
        'bal_accuracy': bal_accuracy,
        'precision':    prec_m,
        'recall':       rec_m,
        'specificity':  specificity,
        'f1_score':     f1_score,
        'auc':          auc,
        'TPr':          tp_m/len(test_outs),
        'FPr':          fp_m/len(test_outs),
        'TNr':          tn_m/len(test_outs),
        'FNr':          fn_m/len(test_outs),
    }

def DoC_feat(val_outs, test_outs, value,):
    # Calculate mean confidence on validaiton
    val_outs = [out if out >= 0.5 else 1 - out for out in val_outs]
    val_conf = np.mean(val_outs)

    # Calculate mean confidence on test
    test_outs = [out if out >= 0.5 else 1 - out for out in test_outs]
    test_conf = np.mean(test_outs)

    # Calculate DoC
    doc = val_conf - test_conf

    return value - doc

def calculate_DoC_metrics(y_pred_validation, y_labels_validation, y_pred_test_probs, is_multilabel=False):
    realized_val_metrics = calculate_metrics(y_labels_validation, y_pred_validation, is_multilabel=is_multilabel)

    # Calculate DoC metrics
    accuracy = DoC_feat(y_pred_validation, y_pred_test_probs, realized_val_metrics['accuracy'])
    bal_accuracy = DoC_feat(y_pred_validation, y_pred_test_probs, realized_val_metrics['bal_accuracy'])
    precision = DoC_feat(y_pred_validation, y_pred_test_probs, realized_val_metrics['precision'])
    recall = DoC_feat(y_pred_validation, y_pred_test_probs, realized_val_metrics['recall'])
    specificity = DoC_feat(y_pred_validation, y_pred_test_probs, realized_val_metrics['specificity'])
    f1 = DoC_feat(y_pred_validation, y_pred_test_probs, realized_val_metrics['f1_score'])
    auroc = DoC_feat(y_pred_validation, y_pred_test_probs, realized_val_metrics['auc'])
    TPr = DoC_feat(y_pred_validation, y_pred_test_probs, realized_val_metrics['TPr'])
    FPr = DoC_feat(y_pred_validation, y_pred_test_probs, realized_val_metrics['FPr'])
    TNr = DoC_feat(y_pred_validation, y_pred_test_probs, realized_val_metrics['TNr'])
    FNr = DoC_feat(y_pred_validation, y_pred_test_probs, realized_val_metrics['FNr'])
    return {
        'accuracy': accuracy,
        'bal_accuracy': bal_accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'auc': auroc,
        'f1_score': f1,
        'TPr': TPr,
        'FPr': FPr,
        'TNr': TNr,
        'FNr': FNr,
    }

# CBPE
def calculate_CBPE_metrics(y_pred_probs, prevalence_correction=False):
    accuracy = CBPE_accuracy(y_pred_probs, prevalence_correction=prevalence_correction)
    bal_accuracy = CBPE_accuracy(y_pred_probs, balanced=True)
    precision = CBPE_precision(y_pred_probs, prevalence_correction=prevalence_correction)
    recall = CBPE_recall(y_pred_probs)
    specificity = CBPE_specificity(y_pred_probs)
    f1 = CBPE_F1(y_pred_probs, prevalence_correction=prevalence_correction)
    auroc = CBPE_auroc(y_pred_probs, show_plots=False)
    TP, FP, TN, FN = CBPE_confusion_matrix(y_pred_probs)

    return {
        'accuracy': accuracy,
        'bal_accuracy': bal_accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'auc': auroc,
        'f1_score': f1,
        'TPr': TP/ len(y_pred_probs),
        'FPr': FP/ len(y_pred_probs),
        'TNr': TN/ len(y_pred_probs),
        'FNr': FN/ len(y_pred_probs),

    }

# ATC
def ATC_metric_estim(val_outs, test_outs, value, threshold=0.5, debug=False):
    s_val = [val_s[0] if val_s >= threshold else 1 - val_s[0] for val_s in val_outs]
    s_test = [test_s[0] if test_s >= threshold else 1 - test_s[0] for test_s in test_outs]

    # print(acc)
    # threshs = np.linspace(0.5, 1, 1000)
    # min_diff = 1e5
    # best_t = 0
    # for t in threshs:
    #     counter = np.sum(np.where(s_val <= t, 1, 0))
    #     diff = np.abs(counter/len(s_val) - value)
    #     if diff < min_diff:
    #         min_diff = diff
    #         best_t = t
    
    best_t = np.percentile(s_val, (1-value) * 100)
    acc_pred_test = np.sum(np.where(s_test >= best_t, 1, 0)) / len(s_test)
    
    if debug:
        print(f'ATC metric: {value}')
        print('best_t: ', best_t, ' with ', np.sum(np.where(s_val >= best_t, 1, 0))/len(s_val))
        print('acc_pred_test: ', acc_pred_test)
        plt.hist(s_val, density=False, label=f'val acc: {value}')
        plt.vlines(best_t, 0,len(s_val), color='red', label=f'best_t = {best_t}')
        plt.legend()
        plt.show()
        plt.hist(s_test, density=False, label=f'test acc: {acc_pred_test}')
        plt.legend()
        plt.vlines(best_t, 0, len(s_test), color='r')
        plt.show()
    return acc_pred_test

def CM_ATC_metric_estim(val_outs, val_labels, test_outs, threshold=0.5, prevalence_correction=False):
     # Val confusion matrix elements
    TP = np.sum(val_outs[val_labels == 1] >= threshold)
    TN = np.sum(val_outs[val_labels == 0] < threshold)
    FP = np.sum(val_outs[val_labels == 0] >= threshold)
    FN = np.sum(val_outs[val_labels == 1] < threshold)
    # print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')
    
    if prevalence_correction:
        P_val = np.mean(val_labels)
        est_P = estimate_prior_em(test_outs)
        if np.abs(est_P - P_val) < 0.1:
            est_P = 0.5

    s_val_0 = [val_s[0] for val_s in val_outs if val_s < threshold]
    s_val_1 = [val_s[0] for val_s in val_outs if val_s >= threshold]

    s_test_0 = [test_s[0] for test_s in test_outs if test_s < threshold]
    s_test_1 = [test_s[0] for test_s in test_outs if test_s >= threshold]

    t_0 = np.percentile(s_val_0, TN/len(s_val_0)*100)
    t_1 = np.percentile(s_val_1, FP/len(s_val_1)*100)
    TP_pred_test = np.sum(np.where(s_test_1 >= t_1, 1, 0)) #/ len(s_test_1)
    TN_pred_test = np.sum(np.where(s_test_0 < t_0, 1, 0)) #/ len(s_test_0)
    FP_pred_test = len(s_test_1) - TP_pred_test #/ len(s_test_0)
    FN_pred_test = len(s_test_0) - TN_pred_test #/ len(s_test_1)

    

    # 1) Prevalence-adjusted precision (as you already have)
    if prevalence_correction:
        precision = est_P * TP_pred_test \
                    / (est_P * TP_pred_test + (1 - est_P) * FP_pred_test)
    else:
        precision = TP_pred_test / (TP_pred_test + FP_pred_test) if (TP_pred_test + FP_pred_test) > 0 else 0
    # 2) Recall (unchanged)
    recall = TP_pred_test / (TP_pred_test + FN_pred_test)

    # 3) Specificity (unchanged)
    specificity = TN_pred_test / (TN_pred_test + FP_pred_test) if (TN_pred_test + FP_pred_test) > 0 else 0

    # 4) Accuracy adjusted to true prevalence
    #    = P * Sensitivity + (1-P) * Specificity
    if prevalence_correction:
        accuracy = est_P * recall + (1 - est_P) * specificity
    else:
        accuracy = (TP_pred_test + TN_pred_test) / (TP_pred_test + TN_pred_test + FP_pred_test + FN_pred_test) if (TP_pred_test + TN_pred_test + FP_pred_test + FN_pred_test) > 0 else 0

    # 5) Balanced accuracy (still the average of the two rates)
    bal_acc = 0.5 * (recall + specificity)

    # 6) F1 uses the adjusted precision
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # plt.hist(s_val_0, density=False, )
    # plt.text(0.5, 4000, f'TP: {TP/len(val_outs)}', fontsize=10)
    # plt.text(0.5, 3500, f'TN: {TN/len(val_outs)}', fontsize=10)
    # plt.text(0.5, 3000, f'FP: {FP/len(val_outs)}', fontsize=10)
    # plt.text(0.5, 2500, f'FN: {FN/len(val_outs)}', fontsize=10)
    # plt.vlines(t_0, 0, 500, color='red', label=f't_0 = {t_0}')
    # plt.hist(s_val_1, density=False, label=f'val acc: {TN/(TN+FP)}')
    # plt.vlines(t_1, 0, 500, color='green', label=f't_1 = {t_1}')
    
    # plt.show()
    # plt.hist(s_test_0, density=False, label=f'test acc: {TP_pred_test/(TP_pred_test+FN_pred_test)}')
    # plt.vlines(t_0, 0, 500, color='red', label=f't_0 = {t_0}')
    # plt.hist(s_test_1, density=False, label=f'test acc: {TN_pred_test/(TN_pred_test+FP_pred_test)}')
    # plt.vlines(t_1, 0, 500, color='green', label=f't_1 = {t_1}')
    # plt.text(0.5, 400, f'TP: {TP_pred_test}', fontsize=10)
    # plt.text(0.5, 350, f'TN: {TN_pred_test}', fontsize=10)
    # plt.text(0.5, 300, f'FP: {FP_pred_test}', fontsize=10)
    # plt.text(0.5, 250, f'FN: {FN_pred_test}', fontsize=10)
    # plt.show()
    # auc
    est_cov_mat = {'TP': TP_pred_test, 'TN': TN_pred_test, 'FP': FP_pred_test, 'FN': FN_pred_test}
    tpr = []
    fpr = []
    thresholds_ = np.linspace(0.001, 0.999, 100)
    thresholds = np.quantile(test_outs, q=thresholds_)
    thresholds = np.unique(thresholds)
    for thresh in thresholds:
        TP = np.sum(val_outs[val_labels == 1] >= thresh)
        TN = np.sum(val_outs[val_labels == 0] < thresh)
        FP = np.sum(val_outs[val_labels == 0] >= thresh)
        FN = np.sum(val_outs[val_labels == 1] < thresh)

        s_val_0 = np.array([val_s[0] for val_s in val_outs if val_s < thresh])
        s_val_1 = np.array([val_s[0] for val_s in val_outs if val_s >= thresh])

        s_test_0 = np.array([test_s[0] for test_s in test_outs if test_s < thresh])
        s_test_1 = np.array([test_s[0] for test_s in test_outs if test_s >= thresh])


        t_0 = np.percentile(s_val_0, TN/len(s_val_0)*100) if len(s_val_0) > 0 else 0.
        t_1 = np.percentile(s_val_1, FP/len(s_val_1)*100) if len(s_val_1) > 0 else 0

        TP_pred_test = np.sum(np.where(s_test_1 >= t_1, 1, 0)) #/ len(s_test_1)
        TN_pred_test = np.sum(np.where(s_test_0 < t_0, 1, 0)) #/ len(s_test_0)
        FP_pred_test = len(s_test_1) - TP_pred_test #/ len(s_test_0)
        FN_pred_test = len(s_test_0) - TN_pred_test #/ len(s_test_1)
        
        tpr_l = TP_pred_test / (TP_pred_test + FN_pred_test) if (TP_pred_test + FN_pred_test) > 0 else 0
        fpr_l = FP_pred_test / (FP_pred_test + TN_pred_test) if (FP_pred_test + TN_pred_test) > 0 else 0
        tpr.append(tpr_l)
        fpr.append(fpr_l)

    auc = np.trapz(tpr[::-1], fpr[::-1])
    
    return {
        'accuracy': accuracy,
        'bal_accuracy': bal_acc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'auc': auc,
        'f1_score': f1_score,
        'TPr': est_cov_mat['TP']/ len(test_outs),
        'FPr': est_cov_mat['FP']/ len(test_outs),
        'TNr': est_cov_mat['TN']/ len(test_outs),
        'FNr': est_cov_mat['FN']/ len(test_outs),
    }#, est_cov_mat

def calculate_ATC_metrics(y_pred_validation, y_labels_validation, y_pred_test_probs, is_multilabel=False):
    realized_val_metrics = calculate_metrics(y_labels_validation, y_pred_validation, is_multilabel=is_multilabel)

    # Calculate ATC metrics
    accuracy = ATC_metric_estim(y_pred_validation, y_pred_test_probs, realized_val_metrics['accuracy'])
    bal_accuracy = ATC_metric_estim(y_pred_validation, y_pred_test_probs, realized_val_metrics['bal_accuracy'])
    precision = ATC_metric_estim(y_pred_validation, y_pred_test_probs, realized_val_metrics['precision'])
    recall = ATC_metric_estim(y_pred_validation, y_pred_test_probs, realized_val_metrics['recall'])
    specificity = ATC_metric_estim(y_pred_validation, y_pred_test_probs, realized_val_metrics['specificity'])
    f1 = ATC_metric_estim(y_pred_validation, y_pred_test_probs, realized_val_metrics['f1_score'])
    auroc = ATC_metric_estim(y_pred_validation, y_pred_test_probs, realized_val_metrics['auc'])
    TPr = ATC_metric_estim(y_pred_validation, y_pred_test_probs, realized_val_metrics['TPr'])
    FPr = ATC_metric_estim(y_pred_validation, y_pred_test_probs, realized_val_metrics['FPr'])
    TNr = ATC_metric_estim(y_pred_validation, y_pred_test_probs, realized_val_metrics['TNr'])
    FNr = ATC_metric_estim(y_pred_validation, y_pred_test_probs, realized_val_metrics['FNr'])
    return {
        'accuracy': accuracy,
        'bal_accuracy': bal_accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'auc': auroc,
        'f1_score': f1,
        'TPr': TPr,
        'FPr': FPr,
        'TNr': TNr,
        'FNr': FNr,
    }





