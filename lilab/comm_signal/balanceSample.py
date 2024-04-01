# from lilab.comm_signal.balanceSample import balanceSample
import numpy as np


def balanceSample(labels, by='percentile', by_value=40):
    assert by in ['percentile', 'number']
    labels_u, labels_c = np.unique(labels, return_counts=True)
    if by=='percentile':
        chooseNum_ = int(np.percentile(labels_c, 40)) * np.ones(len(labels_u))
    elif by=='number':
        chooseNum_ = by_value * np.ones(len(labels_u))
    else:
        raise ValueError('by must be percentile or number')
    
    chooseNum_l = chooseNum_.astype(int)
    np.random.seed(0)
    inds_chosen=[(np.random.choice(a=np.argwhere(labels==lu)[:,0], size=chooseNum, replace=False) 
                if (labels==lu).sum()>chooseNum else np.argwhere(labels==lu)[:,0])
                for chooseNum, lu in zip(chooseNum_l, labels_u)]
    inds_chosen=np.concatenate(inds_chosen)
    labels_balanced=labels[inds_chosen]
    ind_no_chosen = np.ones(len(labels), dtype=bool)
    ind_no_chosen[inds_chosen] = False
    ind_no_chosen = np.where(ind_no_chosen)[0]
    return labels_balanced, inds_chosen, ind_no_chosen

