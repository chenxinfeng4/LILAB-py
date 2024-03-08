# python -m lilab.outlier_refine.s0_errorframe_separate_by_keyboard DIR \\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\multiview_9\chenxf\WTxWT_230724
# %%
import cv2
import os
import os.path as osp
import glob
import argparse
import shutil
import tqdm
import os
#import s0_errorframe_separate_by_keyboard
#from google.colab.patches import cv2_imshow

dir = r'\\liying.cibr.ac.cn\Data_Temp\Chenxinfeng\multiview-large\TPH2KOxWT\tmp_video\frames_overlap'
maximgsize = [1280, 800]

#maximgsize = [2400, 2400]
# use cv2 to read the , image and then imshow
def label(dir):
    imgs = glob.glob(osp.join(dir, '*.png'))
    cv2.namedWindow('TEST', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    for imgpath in tqdm.tqdm(imgs):
        img = cv2.imread(imgpath)
        imgbasename = osp.basename(imgpath)
        

        # resize the image and keep the aspect ratio
        if img.shape[0] > maximgsize[0] or img.shape[1] > maximgsize[1]:
            img = cv2.resize(img, (maximgsize[1], maximgsize[0])) # (width, height)

        cv2.putText(img, imgbasename, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('TEST', img)
        key = cv2.waitKey()
        if key == ord('q'):
            break
        # assert key is alpha-numeric
        assert key in range(48, 58) or key in range(65, 91) or key in range(97, 123)
        # convert to ascii
        key = chr(key)
        # create a folder
        folder = osp.join(dir, key)
        os.makedirs(folder, exist_ok=True)
        # move the image to the folder
        shutil.move(imgpath, folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    args = parser.parse_args()
    assert osp.isdir(args.dir)
    label(args.dir)