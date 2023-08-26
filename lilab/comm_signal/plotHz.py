import numpy as np
import matplotlib.pyplot as plt


def plotHz(Fs, data_l, *args, **kwargs):
    plt.plot(np.arange(len(data_l))/Fs, data_l, *args, **kwargs)
