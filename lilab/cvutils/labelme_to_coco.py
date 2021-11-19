# !pyinstaller -F labelme_to_coco.py -i labelme_to_coco.ico
# chenxinfeng
# ------使用方法------
# 直接拖动文件夹到EXE中
import labelme2coco
import os
import sys
# set directory that contains labelme annotations and image files

def converter(labelme_folder):
    parent_folder, cur_folder = os.path.split(os.path.abspath(labelme_folder))
    
    # set path for coco json to be saved
    save_json_path = os.path.join(parent_folder, cur_folder+"_trainval.json")

    # convert labelme annotations to coco
    labelme2coco.convert(labelme_folder, save_json_path)

if __name__ == '__main__':
    n = len(sys.argv)
    if n == 1:
        folder = input("Select a folder >> ")
        if folder == None:
            exit()
        else:
            sys.argv.append(folder)
            
    print(sys.argv[1:])
    
    labelme_folder = sys.argv[1]
    converter(labelme_folder)
    print("Succeed")
