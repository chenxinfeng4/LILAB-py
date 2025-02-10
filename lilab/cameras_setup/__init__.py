# from lilab.cameras_setup import get_view_xywh_wrapper

from ._get_calibinfo import *
from typing import List
import os.path as osp
import json

current_dir = osp.dirname(osp.abspath(__file__))
def get_view_xywh_wrapper(setupname:str) -> List:
    json_file = osp.join(current_dir, str(setupname), 'views_xywh.json')
    assert osp.exists(json_file), '%s not exists. Maybe not registered'%json_file
    views_xywh = json.load(open(json_file, 'r'))
    return views_xywh


def __create_default_linkview_json(nviews):
    minimal_tree = list(zip(range(nviews-1), range(1,nviews)))
    minimal_tree.append((0, nviews-1))
    linkview = {
        "views": list(range(nviews)),
        "minimal_tree": minimal_tree
    }
    return linkview


def get_json_wrapper(setupname:str):
    json_file = osp.join(current_dir, setupname, 'intrinsics_calib.json')
    assert osp.exists(json_file), '%s not exists. Maybe not registered'%json_file
    intrinsics = json.load(open(json_file, 'r'))

    linkview_file = osp.join(current_dir, setupname, 'link_views.json')
    if osp.exists(linkview_file):
        linkview = json.load(open(linkview_file, 'r'))
    else:
        linkview = __create_default_linkview_json(len(intrinsics))
    return linkview, intrinsics
