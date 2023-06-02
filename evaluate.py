import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score,roc_curve
from sklearn.metrics import precision_recall_fscore_support as prf
from scipy.stats import rankdata
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_highest_scores(scores, true_labels,threshold_steps=500):
    scores_sorted_index = rankdata(scores,'ordinal')
    th_vals = np.array(range(threshold_steps)) * 1.0 / threshold_steps
    fscores = [None]*threshold_steps
    thresholds = [None]*threshold_steps
    for i in tqdm(range(threshold_steps)):
        cur_pred = scores_sorted_index > th_vals[i] * len(scores)
        fscores[i] = f1_score(true_labels, cur_pred)
        score_index = scores_sorted_index.tolist().index(int(th_vals[i] * len(scores))+1)
        thresholds[i] = scores[score_index]

    top_fscore = max(fscores)
    threshold = thresholds[fscores.index(top_fscore)]

    pred_labels = np.zeros(len(scores))
    pred_labels[scores > threshold] = 1
    # for i in range(len(pred_labels)):
    #     pred_labels[i] = int(pred_labels[i])
    #     true_labels[i] = int(true_labels[i])
    pre = precision_score(true_labels, pred_labels)
    rec = recall_score(true_labels, pred_labels)
    fscore = 2*pre*rec/(pre+rec)
    auc_score = roc_auc_score(true_labels, scores)
    return fscore, pre, rec, auc_score


def get_scores(scores, labels, ratio):
    thresh = np.percentile(scores, ratio)
    y_pred = (scores >= thresh).astype(int)
    y_true = labels.astype(int)
    precision, recall, f_score, support = prf(y_true, y_pred, average='binary')
    auc_score = roc_auc_score(labels, scores)
    return f_score, precision, recall, auc_score


def plot_roc_curve(labels, scores):
    fpr, tpr, _ = roc_curve(labels,scores)
    plt.plot(fpr,tpr,color='red')
    plt.plot([0,1],[0,1],linestyle='dashed', color='k')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()


def plot_roc_curve_inout(labels_in, scores_in, labels_out, scores_out):
    fpr, tpr, _ = roc_curve(labels_in,scores_in)
    plt.plot(fpr,tpr,color='red',label='Inlier')
    fpr, tpr, _ = roc_curve(labels_out, scores_out)
    plt.plot(fpr, tpr, color='blue', label='Outlier')
    plt.plot([0,1],[0,1],linestyle='dashed', color='k')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.title('ROC')
    plt.legend()
    plt.show()