import glob
import os
from PIL import Image

def concat(folder_1, folder_2):
    folder_out = folder_1 + '_x_' + os.path.split(folder_2)[-1]
    os.makedirs(folder_out, exist_ok=True)

    def globimages(folder):
        ret =   glob.glob(os.path.join(folder, '*.jpg')) + \
                glob.glob(os.path.join(folder, '*.jpeg')) + \
                glob.glob(os.path.join(folder, '*.jpeg'))
        return ret

    def commonimages(imgs1, imgs2):
        imgsNake1 = [os.path.split(img)[-1] for img in imgs1]
        imgsNake2 = [os.path.split(img)[-1] for img in imgs2]
        imgsNakeCommon = set(imgsNake1) & set(imgsNake2)
        imgsCommon1 = [img for img, imgNake in zip(imgs1, imgsNake1)
                             if imgNake in imgsNakeCommon]
        imgsCommon2 = [img for img, imgNake in zip(imgs2, imgsNake2)
                             if imgNake in imgsNakeCommon]
        return imgsCommon1, imgsCommon2

    imgs1 = globimages(folder_1)
    imgs2 = globimages(folder_2)
    imgs1, imgs2 = commonimages(imgs1, imgs2)

    for img1, img2 in zip(imgs1, imgs2):
        images = [Image.open(x) for x in [img1, img2]]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]

        new_im.save(os.path.join(folder_out, os.path.split(img1)[-1]))
    
if __name__ == '__main__':
    folder_1 = input("(1/2)The left  folder: >>")
    folder_2 = input("(2/2)The right folder: >>")
    concat(folder_1, folder_2)