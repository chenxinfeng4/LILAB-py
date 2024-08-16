# python -m lilab.cvutils.concat_image_vstack A/ B/
import glob
import os
import os.path as osp
import tqdm
from PIL import Image
import argparse


def concat(folder_1, folder_2, folder_3):
    folder_out = (
        osp.abspath(folder_1)
        + "_x_"
        + os.path.split(osp.abspath(folder_2))[-1]
        + "_x_"
        + os.path.split(osp.abspath(folder_3))[-1]
    )
    os.makedirs(folder_out, exist_ok=True)

    def globimages(folder):
        ret = (
            glob.glob(os.path.join(folder, "*.jpg"))
            + glob.glob(os.path.join(folder, "*.jpeg"))
            + glob.glob(os.path.join(folder, "*.png"))
        )
        return ret

    def commonimages(imgs1, imgs2, imgs3):
        imgsNake1 = [os.path.split(img)[-1] for img in imgs1]
        imgsNake2 = [os.path.split(img)[-1] for img in imgs2]
        imgsNake3 = [os.path.split(img)[-1] for img in imgs3]
        imgsNakeCommon = set(imgsNake1) & set(imgsNake2) & set(imgsNake3)
        imgsCommon1 = [
            img for img, imgNake in zip(imgs1, imgsNake1) if imgNake in imgsNakeCommon
        ]
        imgsCommon2 = [
            img for img, imgNake in zip(imgs2, imgsNake2) if imgNake in imgsNakeCommon
        ]
        imgsCommon3 = [
            img for img, imgNake in zip(imgs3, imgsNake3) if imgNake in imgsNakeCommon
        ]
        return imgsCommon1, imgsCommon2, imgsCommon3

    imgs1 = globimages(folder_1)
    imgs2 = globimages(folder_2)
    imgs3 = globimages(folder_3)
    imgs1, imgs2, imgs3 = commonimages(imgs1, imgs2, imgs3)

    for img1, img2, img3 in zip(tqdm.tqdm(imgs1), imgs2, imgs3):
        images = [Image.open(x) for x in [img1, img2, img3]]
        widths, heights = zip(*(i.size for i in images))

        total_heights = sum(heights)
        max_width = max(widths)

        new_im = Image.new("RGB", (max_width, total_heights))

        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]

        new_im.save(os.path.join(folder_out, os.path.split(img1)[-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_1", type=str)
    parser.add_argument("folder_2", type=str)
    parser.add_argument("folder_3", type=str)
    args = parser.parse_args()
    concat(args.folder_1, args.folder_2, args.folder_3)
