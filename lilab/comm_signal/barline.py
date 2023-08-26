import numpy as np
import matplotlib.pyplot as plt
import warnings


def barline(x:list, HL:list=None, color='k', *args, **kwargs):
    """
    Replicates the functionality of the barline_old() function in MATLAB.

    Input parameters:
    x : 1D array representing the positions of the bars
    HL: a 2-element tuple or list containing the high and low values for the plot
    *args: additional plotting parameters passed to 'plot()'

    Output parameters:
    hline: handle for plot line, 1x1
    """
    # Prepare input arguments
    if HL is None:
        HL = [0, 1]
    elif type(HL) in [int, float]:
        HL = [HL, HL + 1]
    elif len(HL) != 2:
        raise ValueError("Entry a number")

        
    HL = np.array(HL)
    x = np.array(x)

    if x.size == 0:
        warnings.warn("barpatch for empty data")
        return None

    assert x.ndim == 1, "x must be 1-D array"

    # Create data
    xtick = np.concatenate([[x0, x0, np.nan, np.nan] for x0 in x])
    ytick = np.concatenate([[*HL, np.nan, np.nan] for x0 in x])

    # Plot data
    hline, = plt.plot(xtick, ytick, color=color, *args, **kwargs)
    return hline
