# NFRAME = 14
NFRAME = 4
_nclass = 2
_njoint = 14
out_numpy_imgNKHW_shape = [9,_nclass,800,1280]
out_numpy_com2d_shape = [9,_nclass,2]
out_numpy_previ_shape = [800,1280,3]
out_numpy_timecode_shape = 6

dannce_numpy_X_shape = [_nclass, 64, 64, 64, 9]
dannce_numpy_Xgrid_shape = [_nclass, 64, 64, 64, 3]
ifshow = True

import multiprocessing
import numpy as np
from typing import List
import ctypes

def create_shared_arrays() -> List[multiprocessing.Array]:
    shared_array_imgNKHW = multiprocessing.Array('b', int(NFRAME*np.prod(out_numpy_imgNKHW_shape)))  #np.uint8
    shared_array_com2d = multiprocessing.Array('d', int(NFRAME*np.prod(out_numpy_com2d_shape)))      #np.float64
    shared_array_previ = multiprocessing.Array('b', int(NFRAME*np.prod(out_numpy_previ_shape)))      #np.uint8
    shared_array_timecode = multiprocessing.Array('d', int(NFRAME*out_numpy_timecode_shape)) #np.float64
    return shared_array_imgNKHW, shared_array_com2d, shared_array_previ, shared_array_timecode

def get_numpy_handle(shared_array_imgNKHW, shared_array_com2d, shared_array_previ, shared_array_timecode) -> List[np.ndarray]:
    numpy_imgNKHW = np.frombuffer(shared_array_imgNKHW.get_obj(), dtype=np.uint8).reshape((NFRAME,*out_numpy_imgNKHW_shape))
    numpy_com2d = np.frombuffer(shared_array_com2d.get_obj(), dtype=np.float64).reshape((NFRAME, *out_numpy_com2d_shape))
    numpy_previ = np.frombuffer(shared_array_previ.get_obj(), dtype=np.uint8).reshape((NFRAME, *out_numpy_previ_shape))
    numpy_timecode = np.frombuffer(shared_array_timecode.get_obj(), dtype=np.float64).reshape((NFRAME, out_numpy_timecode_shape)) #(timecode, delay1, delay2, delay3)
    return numpy_imgNKHW, numpy_com2d, numpy_previ, numpy_timecode

def create_shared_arrays_dannce() -> List[multiprocessing.Array]:
    share_dannce_X = multiprocessing.Array(ctypes.c_int32, int(NFRAME*np.prod(dannce_numpy_X_shape))) #np.float32
    share_dannce_Xgrid = multiprocessing.Array(ctypes.c_int16, int(NFRAME*np.prod(dannce_numpy_Xgrid_shape))) #np.float16
    share_dannce_imgpreview = multiprocessing.Array(ctypes.c_ubyte, int(NFRAME*np.prod(out_numpy_previ_shape))) #np.uint8
    return share_dannce_X, share_dannce_Xgrid, share_dannce_imgpreview

def get_numpy_handle_dannce(share_dannce_X, share_dannce_Xgrid, share_dannce_imgpreview) -> List[np.ndarray]:
    numpy_dannce_X = np.frombuffer(share_dannce_X.get_obj(), dtype=np.float32).reshape((NFRAME, *dannce_numpy_X_shape))
    numpy_dannce_Xgrid = np.frombuffer(share_dannce_Xgrid.get_obj(), dtype=np.float16).reshape((NFRAME, *dannce_numpy_Xgrid_shape))
    numpy_dannce_imgpreview = np.frombuffer(share_dannce_imgpreview.get_obj(), dtype=np.uint8).reshape((NFRAME, *out_numpy_previ_shape))
    return numpy_dannce_X, numpy_dannce_Xgrid, numpy_dannce_imgpreview