# from lilab.comm_signal.plotSVM import plotSVM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def fold_5_f1(X1:np.ndarray, X2:np.ndarray, method='SVM'):
    """
    calcuate f1 score using leave one out cross validation
    """
    assert X1.shape[1] == X2.shape[1] == 2
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(len(X1)), np.ones(len(X2))))
    y_l = []
    loo = LeaveOneOut()
    
    if method == 'SVM':
        clf = SVC(kernel='linear', C=10**4)
    elif method == 'LDA':
        clf = LinearDiscriminantAnalysis()
    else:
        raise NameError()
    
    for fold, (train_indices, val_indices) in enumerate(loo.split(X, y)):
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        y_l.append([y_val_pred, y_val])

    y_l = np.concatenate(y_l,axis=-1)
    y_pred, y = y_l
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return accuracy, f1


def fold_5_f1_C3(X1:np.ndarray, X2:np.ndarray, X3:np.ndarray, method='SVM'):
    """
    calcuate f1 score using leave one out cross validation
    """
    assert X1.shape[1] == X2.shape[1] == X3.shape[1] == 2
    X = np.vstack((X1, X2, X3))
    y = np.hstack((np.zeros(len(X1)), np.ones(len(X2)), np.ones(len(X3))*2))
    y_l = []
    loo = LeaveOneOut()

    if method == 'SVM':
        clf = SVC(kernel='linear', C=10**4)
    elif method == 'LDA':
        clf = LinearDiscriminantAnalysis()
    else:
        raise NameError()
    
    for fold, (train_indices, val_indices) in enumerate(loo.split(X, y)):
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        y_l.append([y_val_pred, y_val])
   
    y_l = np.concatenate(y_l,axis=-1)
    y_pred, y = y_l
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    return accuracy, f1



def plotSVM(X1:np.ndarray, X2:np.ndarray, use_crossValid=False):
    assert X1.shape[1] == X2.shape[1] == 2
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(len(X1)), np.ones(len(X2))))

    clf = SVC(kernel='linear', C=10**5)  # C 大于 10^6 时，模型会过拟合
    clf.fit(X, y)
    if use_crossValid:
        accuracy, f1 = fold_5_f1(X1, X2)
    else:
        y_pred = clf.predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
    xlim = plt.xlim()
    ylim = plt.ylim()
    top_right_corner = np.array([xlim[1], ylim[1]])
    # plot the text of accuracy in the corner, align to the right
    plt.text(top_right_corner[0] , top_right_corner[1], 
             f'\nAccuracy: {accuracy:.2f} F1: {f1:.2f}', fontsize=10, color='red', ha='right', va='top')
    
    # 创建网格来评估模型
    xx = np.linspace(xlim[0], xlim[1], 120)
    yy = np.linspace(ylim[0], ylim[1], 120)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    Z = clf.predict(xy).reshape(XX.shape)
    # 绘制决策边界和边界
    plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])


def plotSVM3(X1:np.ndarray, X2:np.ndarray, X3:np.ndarray, use_crossValid=False):
    assert X1.shape[1] == X2.shape[1] == X3.shape[1] == 2
    X = np.vstack((X1, X2, X3))
    y = np.hstack((np.zeros(len(X1)), np.ones(len(X2)), np.ones(len(X3))+1))
    clf = SVC(kernel='linear', C=10**5)  # C 大于 10^6 时，模型会过拟合
    clf.fit(X, y)
    if use_crossValid:
        accuracy, f1 = fold_5_f1_C3(X1, X2, X3)
    else:
        y_pred = clf.predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='macro')
    
    xlim = plt.xlim()
    ylim = plt.ylim()
    top_right_corner = np.array([xlim[1], ylim[1]])
    # plot the text of accuracy in the corner, align to the right
    plt.text(top_right_corner[0] , top_right_corner[1],
             f'\nAccuracy: {accuracy:.2f} F1: {f1:.2f}', fontsize=10, color='red', ha='right', va='top')

    # 创建网格来评估模型
    xx = np.linspace(xlim[0], xlim[1], 120)
    yy = np.linspace(ylim[0], ylim[1], 120)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = np.argmax(clf.decision_function(xy), axis=-1).reshape(XX.shape)
    Z = clf.predict(xy).reshape(XX.shape)

    # 绘制决策边界和边界
    plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1, 2], alpha=0.5, linestyles=['--', '--', '--', '--'])



def plotSVM_zzcuse(X1:np.ndarray, X2:np.ndarray, use_crossValid=False):
    assert X1.shape[1] == X2.shape[1] == 2
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(len(X1)), np.ones(len(X2))))

    clf = SVC(kernel='linear', C=10**5)  # C 大于 10^6 时，模型会过拟合
    clf.fit(X, y)
    if use_crossValid:
        accuracy, f1 = fold_5_f1(X1, X2)
    else:
        y_pred = clf.predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
    xlim = plt.xlim()
    ylim = plt.ylim()
    top_right_corner = np.array([xlim[1], ylim[1]])
    # plot the text of accuracy in the corner, align to the right
    # plt.text(top_right_corner[0] , top_right_corner[1], 
    #          f'\nAccuracy: {accuracy:.2f} F1: {f1:.2f}', fontsize=10, color='red', ha='right', va='top')
    
    # # 创建网格来评估模型
    # xx = np.linspace(xlim[0], xlim[1], 120)
    # yy = np.linspace(ylim[0], ylim[1], 120)
    # YY, XX = np.meshgrid(yy, xx)
    # xy = np.vstack([XX.ravel(), YY.ravel()]).T
    # Z = clf.decision_function(xy).reshape(XX.shape)
    # Z = clf.predict(xy).reshape(XX.shape)
    # # 绘制决策边界和边界
    # plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    return accuracy, f1