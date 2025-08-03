# from lilab.comm_signal.binmean import binmean
import numpy as np

def binmean(dataIn:np.ndarray, binsize:int=4) -> np.ndarray:
    """
    %对一段数据以binsize逐段取平均
    %function data_1y = binmean(data, binsize)
    %----Input 参数---------
    % data            :  数据, 1v 或 ntrial_x_nsample
    % binsize         :  bin的大小, 1num
    %
    %----Output 参数---------
    % data_bin         :  平均后的数据
    """
    dataIn = np.array(dataIn)
    assert dataIn.shape[-1] >= 2*binsize
    end_point = dataIn.shape[-1] // binsize * binsize
    dataIn1 = dataIn[..., :end_point]
    signal_output = np.mean(dataIn1.reshape(dataIn1.shape[:-1] + (dataIn1.shape[-1]//binsize, binsize)), axis=-1)
    return signal_output
