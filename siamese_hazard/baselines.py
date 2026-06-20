import numpy as np
from sklearn.linear_model import LogisticRegression

def train_baseline_lr(Z0_tr, y_tr, Z0_va):
    clf=LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(Z0_tr,y_tr); return clf, clf.predict_proba(Z0_va)[:,1]

def train_delta_lr(Z0_tr,Z1_tr,y_tr,Z0_va,Z1_va):
    d_tr=Z1_tr-Z0_tr; d_va=Z1_va-Z0_va
    clf=LogisticRegression(max_iter=200, class_weight="balanced")
    clf.fit(np.hstack([Z0_tr,d_tr]), y_tr)
    return clf, clf.predict_proba(np.hstack([Z0_va,d_va]))[:,1]
