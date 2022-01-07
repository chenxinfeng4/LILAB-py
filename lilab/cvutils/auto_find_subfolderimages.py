import os
import os.path as osp
import glob


def find_subfolderimages(dir_or_fileroot, images_name):
    # get the dir of dir_or_fileroot
    dir_or_fileroot = os.path.abspath(dir_or_fileroot)
    assert osp.exists(dir_or_fileroot), '{} does not exist'.format(dir_or_fileroot)
    dir_path = dir_or_fileroot if osp.isdir(dir_or_fileroot) else osp.dirname(dir_or_fileroot)
    images_basename = [osp.basename(image_name) for image_name in images_name]

    # get the subfolder of dir_path
    subfolder_list = [osp.join(dir_path, subfolder) for subfolder in os.listdir(dir_path) 
                      if osp.isdir(osp.join(dir_path, subfolder))]

    # loop through the subfolder
    for subfolder in subfolder_list:
        # list the file in the subfolder
        subs = os.listdir(subfolder)
        if images_basename[0] in subs and images_basename[1] in subs:
            imagefolder = subfolder
            print('{} is the subfolder'.format(subfolder))
            break
    else:
        raise ValueError('{} is not in subfolder'.format(dir_or_fileroot))

    # get the images_name
    full_images_name = [osp.join(imagefolder, image_name) for image_name in images_name]
    return full_images_name
    
