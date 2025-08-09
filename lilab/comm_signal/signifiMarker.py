# from lilab.comm_signal.signifiMarker import signifiMarker
import numpy as np

def fun_marker(p, disp_cutof):
    if p<0.001:
        marker = '***'
    elif p<0.01:
        marker = '**'
    elif p<0.05:
        marker = '*'
    elif disp_cutof is not None and p<disp_cutof:
        marker = f'p={p:.3f}'
    else:
        marker = 'ns'
    return marker


def signifiMarker(pval_l, disp_cutoff=None, disp_ns=True) -> np.ndarray:
    if isinstance(pval_l, float):
        pval_l = [pval_l]
    assert isinstance(pval_l, list) or isinstance(pval_l, np.ndarray)

    marker = np.array([fun_marker(p, disp_cutoff) for p in pval_l])
    if not disp_ns:
        marker[marker=='ns'] = '' 

    if len(pval_l) == 1:
        marker = marker[0]
    return marker
