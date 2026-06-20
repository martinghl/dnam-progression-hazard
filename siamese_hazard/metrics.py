import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

def safe_auc(y_true, y_prob):
    try: return roc_auc_score(y_true, y_prob)
    except: return float("nan")

def safe_pr(y_true, y_prob):
    try: return average_precision_score(y_true, y_prob)
    except: return float("nan")

def brier(y_true, y_prob):
    try: return brier_score_loss(y_true, y_prob)
    except: return float("nan")

def ece_uniform(y_true, y_prob, n_bins=15):
    pt, pp = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return float(np.abs(pt-pp).mean())

def decision_curve(y_true, y_prob, thresholds=None):
    if thresholds is None: thresholds=np.arange(0.05,0.501,0.01)
    y_true=np.asarray(y_true).astype(int); y_prob=np.asarray(y_prob); N=len(y_true)
    out=[]
    for t in thresholds:
        preds=(y_prob>=t).astype(int)
        TP=((preds==1)&(y_true==1)).sum(); FP=((preds==1)&(y_true==0)).sum()
        nb=(TP/N) - (FP/N)*(t/(1-t))
        out.append((t,nb))
    return np.array(out)
