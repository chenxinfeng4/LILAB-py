# python -m lilab.mmlab_scripts.segpkl_smooth A.pkl
# %%
import numpy as np
import mmcv
import copy
import argparse

# %%
def convert(segpkl):
    data = mmcv.load(segpkl)

    assert data, 'data is empty'
    nclasslist = [len(frame[0]) for frame in data]
    nclassset = set(nclasslist)
    assert len(nclassset) == 1, 'The number of classes in each frame is not the same!'

    nclass = nclasslist[0]

    canvas_zero = None
    for frame in data:
        for iclass in range(nclass):
            if frame[1][iclass]:
                shape = frame[1][iclass][0]['size']
                canvas_zero = np.zeros(shape)
                break
        if canvas_zero is not None:
            break
    else:
        raise ValueError('No mask found in the data!')

    # %%
    outdata = mmcv.load(segpkl)
    for iclass in range(nclass):
        valid = [len(frame[0][iclass])>0 for frame in data]
        if np.mean(valid) < 0.5:
            for frame_out in outdata:
                frame_out[0][iclass] = []
                frame_out[1][iclass] = []
            continue
        preboxes = []
        preseg = []
        for frame, frame_out in zip(data+list(reversed(data)), outdata+list(reversed(data))):
            if len(frame[0][iclass])>0:
                preboxes = frame[0][iclass]
                preseg = frame[1][iclass]
            else:
                frame_out[0][iclass] = preboxes
                frame_out[1][iclass] = preseg

    seg_smooth_pkl = segpkl.replace('.pkl', '_smooth.pkl')
    mmcv.dump(outdata, seg_smooth_pkl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('segpkl', type=str, 
                        help='segment pkl file')
    args = parser.parse_args()
    seg_pkl = convert(args.segpkl)
    print('smooth_pkl:', seg_pkl)
