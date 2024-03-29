from ._get_calibinfo import (get_json_1280x800x10, 
    get_json_800x600x6, get_json_1280x800x4,
    get_json_1280x800x9, get_json_bob, get_json_carl,
    get_ballglobal_cm, get_json_zyy)

def get_view_hflip():
    return [0, 1, 0, 
            1, 0, 1, 
            1, 0, 0,
            0]

# from lilab.cameras_setup import get_view_xywh_wraper
def get_view_xywh_wrapper(nviews):
    if nviews==4:
        return get_view_xywh_1280x800x4()
    elif nviews==6:
        return get_view_xywh_800x600x6()
    elif nviews in [9, '9', 'ana', 'bob', 'carl']:
        return get_view_xywh_1280x800x9()
    elif nviews == 'zyy':
        return get_view_xywh_2448x2048x6()
    elif nviews==10:
        return get_view_xywh_1280x800x10()
    elif nviews in ['els']: # early life stress
        return get_view_xywh_800x600x12()
    else:
        raise NotImplementedError

def get_json_wrapper(nviews):
    if nviews==4:
        return get_json_1280x800x4()
    elif nviews==6:
        return get_json_800x600x6()
    elif nviews==9:
        return get_json_1280x800x9()
    elif nviews==10:
        return get_json_1280x800x10()
    elif nviews=='ana':
        return get_json_1280x800x9()
    elif nviews=='bob':
        return get_json_bob()
    elif nviews=='carl':
        return get_json_carl()
    elif nviews=='zyy':
        return get_json_zyy()
    else:
        raise NotImplementedError

def get_view_xywh_1280x800x10():
    w, h = 1280, 800
    crop_xywh = [[w*0,h*0,w,h],
                [w*1,h*0,w,h],
                [w*2,h*0,w,h],
                [w*0,h*1,w,h],
                [w*1,h*1,w,h],
                [w*2,h*1,w,h],
                [w*0,h*2,w,h],
                [w*1,h*2,w,h],
                [w*2,h*2,w,h],
                [w*0,h*3,w,h]]
    return crop_xywh

def get_view_xywh_1280x800x9():
    w, h = 1280, 800
    crop_xywh = [[w*0,h*0,w,h],
                [w*1,h*0,w,h],
                [w*2,h*0,w,h],
                [w*0,h*1,w,h],
                [w*1,h*1,w,h],
                [w*2,h*1,w,h],
                [w*0,h*2,w,h],
                [w*1,h*2,w,h],
                [w*2,h*2,w,h]]
    return crop_xywh

def get_view_xywh_800x600x12():
    w, h = 800, 600
    # crop_xywh = [[w*0,h*0,w,h],
    #             [w*1,h*0,w,h],
    #             [w*2,h*0,w,h],
    #             [w*0,h*1,w,h],
    #             [w*1,h*1,w,h],
    #             [w*2,h*1,w,h],
    #             [w*0,h*2,w,h],
    #             [w*1,h*2,w,h],
    #             [w*2,h*2,w,h],
    #             [w*0,h*3,w,h],
    #             [w*1,h*3,w,h],
    #             [w*2,h*3,w,h]]
    crop_xywh = [[w*2,h*0,w,h],
                [w*2,h*1,w,h],
                [w*2,h*2,w,h],
                [w*2,h*3,w,h]]
    return crop_xywh

def get_view_xywh_1280x800x4():
    w, h = 1280, 800
    crop_xywh = [[w*0,h*0,w,h],
                [w*1,h*0,w,h],
                [w*0,h*1,w,h],
                [w*1,h*1,w,h]]
    return crop_xywh

def get_view_xywh_2448x2048x6():
    w, h = 2448, 2048
    crop_xywh = [[w*0,h*0,w,h],
                [w*1,h*0,w,h],
                [w*2,h*0,w,h],
                [w*0,h*1,w,h],
                [w*1,h*1,w,h],
                [w*2,h*1,w,h]]
    return crop_xywh

def get_view_xywh_800x600x6():
    w, h = 800, 600
    crop_xywh = [[w*0,h*0,w,h],
                [w*1,h*0,w,h],
                [w*2,h*0,w,h],
                [w*0,h*1,w,h],
                [w*1,h*1,w,h],
                [w*2,h*1,w,h]]
    return crop_xywh

def get_view_xywh_640x700x2():
    w, h = 640, 700
    crop_xywh = [[w*0,h*0,w,h],
                [w*1,h*0,w,h]]
    return crop_xywh
