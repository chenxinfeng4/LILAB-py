import os
import os.path as osp
import glob
import json


def find_subfolderimages(dir_or_fileroot, images_name):
    # get the dir of dir_or_fileroot
    dir_or_fileroot = os.path.abspath(dir_or_fileroot)
    assert osp.exists(dir_or_fileroot), "{} does not exist".format(dir_or_fileroot)
    dir_path = (
        dir_or_fileroot if osp.isdir(dir_or_fileroot) else osp.dirname(dir_or_fileroot)
    )
    images_basename = [osp.basename(image_name) for image_name in images_name]

    # get the subfolder of dir_path
    subfolder_list = [
        osp.join(dir_path, subfolder)
        for subfolder in os.listdir(dir_path)
        if osp.isdir(osp.join(dir_path, subfolder))
    ]

    # loop through the subfolder
    for subfolder in subfolder_list:
        # list the file in the subfolder
        subs = os.listdir(subfolder)
        if images_basename[0] in subs and images_basename[1] in subs:
            imagefolder = subfolder
            print("{} is the subfolder".format(subfolder))
            break
    else:
        raise ValueError("{} is not in subfolder".format(dir_or_fileroot))

    # get the images_name
    full_images_name = [osp.join(imagefolder, image_name) for image_name in images_name]
    return full_images_name


def find_coco_subfolderimages(coco_file):
    with open(coco_file, "r") as f:
        coco = json.load(f)

    image_files = set([anno_img["file_name"] for anno_img in coco["images"]])

    import os.path as osp

    parentdir = osp.dirname(coco_file)
    sub_dirs = [
        osp.join(parentdir, sub)
        for sub in os.listdir(parentdir)
        if osp.isdir(osp.join(parentdir, sub))
    ]

    for sub_dir in sub_dirs:
        sub_dir_contains = set(os.listdir(sub_dir))
        if image_files < sub_dir_contains:
            return sub_dir
    else:
        return None


def find_subfolderfiles(dir_or_fileroot, filestofind):
    # get the dir of dir_or_fileroot
    dir_or_fileroot = os.path.abspath(dir_or_fileroot)
    assert osp.exists(dir_or_fileroot), "{} does not exist".format(dir_or_fileroot)
    dir_path = (
        dir_or_fileroot if osp.isdir(dir_or_fileroot) else osp.dirname(dir_or_fileroot)
    )
    file_basename = [osp.basename(filestofind) for filestofind in filestofind]

    # get the subfolder of dir_path
    subfolder_list = [dir_path] + [
        osp.join(dir_path, subfolder)
        for subfolder in os.listdir(dir_path)
        if osp.isdir(osp.join(dir_path, subfolder))
    ]

    # loop through the subfolder
    for subfolder in subfolder_list:
        # list the file in the subfolder
        subs = os.listdir(subfolder)
        if file_basename[0] in subs and file_basename[1] in subs:
            videofolder = subfolder
            print("{} is the subfolder".format(subfolder))
            break
    else:
        raise ValueError("{} is not in subfolder".format(dir_or_fileroot))

    # get the images_name
    file_files_name = [osp.join(videofolder, f) for f in file_basename]
    return file_files_name
