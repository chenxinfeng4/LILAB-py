# python -m lilab.cvutils_new.crop_image a/b/ --setupname carl
# python -m lilab.cvutils_new.crop_image a/b/c.png
import os.path as osp
import os
import argparse
import glob
import cv2
from lilab.cameras_setup import get_view_xywh_wrapper


def xywh2whxy(xywh, keepXeqY=False):
    if keepXeqY:
        maxXY = max(xywh[2:])
        xywh[2] = xywh[3] = maxXY
    whxy = (xywh[2], xywh[3], xywh[0], xywh[1])
    return whxy


def convert(filename, setupname):
    im = cv2.imread(filename)
    views = get_view_xywh_wrapper(setupname)
    outdir = osp.join(osp.dirname(filename), "crop")
    os.makedirs(outdir, exist_ok=True)
    for postfix, crop_xywh in enumerate(views):
        outfile = os.path.splitext(os.path.join(outdir, os.path.basename(filename)))[0]
        outfileformat = "{}_output_" + str(postfix) + ".jpg"
        outfilename = outfileformat.format(outfile)
        # crop the image
        x, y, w, h = crop_xywh
        im1 = im[y : y + h, x : x + w]
        cv2.imwrite(outfilename, im1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop image")
    parser.add_argument(
        "image_path", type=str, default=None, help="path to image or folder"
    )
    parser.add_argument("--setupname", type=str, default="carl")
    args = parser.parse_args()

    image_path = args.image_path
    assert osp.exists(image_path), "image_path not exists"
    if osp.isfile(image_path):
        image_path = [image_path]
    elif osp.isdir(image_path):
        # image_path = [f for f in glob.glob(osp.join(image_path, '*.jpg'))
        #                 if f[-4] not in '0123456789']
        image_path = glob.glob(osp.join(image_path, "*.jpg")) + glob.glob(
            osp.join(image_path, "*.png")
        )
        assert len(image_path) > 0, "no image found"
    else:
        raise ValueError("image_path is not a file or folder")

    for filename in image_path:
        convert(filename, args.setupname)
    print("Succeed")
