# from lilab.comm_signal.line_scale import line_scale
import numpy as np

def line_scale(XR, YR, XNow):
    """
    XR: 1x2 ndarray, the [start, end] value of the input range X
    YR: 1x2 ndarray, the [start, end] value of the output range Y
    XNow: n-dimensional ndarray, the input data

    Returns:
    YNow: n-dimensional ndarray, with same size as XNow
    """
    XR = np.array(XR).squeeze()
    YR = np.array(YR).squeeze()
    if not isinstance(XNow, np.ndarray):
        XNow = np.array(XNow)

    assert XR.shape == YR.shape == (2,)

    assert XR[0]!=XR[1], "The values in XR are identical"

    k = (YR[1] - YR[0]) / (XR[1] - XR[0])
    b = (XR[0] * YR[1] - XR[1] * YR[0]) / (XR[0] - XR[1])
    YNow = k * XNow + b
    return YNow
