import json
import numpy as np

def get_json_1280x800x10():
    setup_json = json.loads(setup_json1280x800x10)
    intrinsics_rational_json = json.loads(intrinsics_rational_json1280x800x10)
    return setup_json, intrinsics_rational_json


def get_json_800x600x6():
    setup_json = json.loads(setup_json800x600x6)
    intrinsics_rational_json = json.loads(intrinsics_rational_json800x600x6)
    return setup_json, intrinsics_rational_json


def get_json_1280x800x4():
    setup_json = json.loads(setup_json1280x800x4)
    intrinsics_rational_json = json.loads(intrinsics_rational_json1280x800x4)
    return setup_json, intrinsics_rational_json


def get_ballglobal_cm():
    fitball_xyz_global =  np.array([[0, 0, 0],
                                [0, 22, 0],
                                [22, 22, 0],
                                [22, 0, 0],
                                [0, 0, 15]], dtype=np.float32)
    return fitball_xyz_global

def get_ballglobal_mm():
    fitball_xyz_global =  np.array([[0, 0, 0],
                                [0, 220, 0],
                                [220, 220, 0],
                                [220, 0, 0],
                                [0, 0, 200]], dtype=np.float32)
    return fitball_xyz_global

setup_json1280x800x10 = """
{
  "views": [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
  "minimal_tree": [
    [9, 0],
    [9, 1],
    [9, 2],
    [9, 3],
    [9, 4],
    [9, 5],
    [9, 6],
    [9, 7],
    [9, 8]
  ]
}
"""


setup_json800x600x6 = """
{
  "views": [ 0, 1, 2, 3, 4, 5],
  "minimal_tree": [
    [1, 0],
    [2, 1],
    [3, 2],
    [4, 3],
    [5, 4],
    [0, 5]
  ]
}
"""

setup_json1280x800x4 = """
{
  "views": [ 0, 1, 2, 3],
  "minimal_tree": [
    [1, 0],
    [2, 1],
    [3, 2]
  ]
}
"""


intrinsics_rational_json1280x800x10 = """
{
    "0": {
        "date": "2021-11-10 21:55:50",
        "description": "",
        "K": [[1120, 0.0, 606],
            [0.0,1183,386],
            [ 0.0, 0.0, 1.0]],
        "K_new": [[1120, 0.0, 606],
            [0.0,1183,386],
            [ 0.0, 0.0, 1.0]],
        "dist": [ 0,0, 0,0, 0,0, 0,0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0],
        "reproj_error": 0.22981242234408908,
        "image_shape": [800, 1280]
    },
    "1": {
        "date": "2021-11-10 21:55:50",
        "description": "",
        "K": [[1157, 0.0, 639.24],
            [0.0,1155.39, 414.51],
            [ 0.0, 0.0, 1.0]],
        "K_new": [[1157, 0.0, 639.24],
            [0.0,1155.39, 414.51],
            [ 0.0, 0.0, 1.0]],
        "dist": [ 0.00, 0.00, 0,0, 0,0, 0,0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0],
        "reproj_error": 0.22981242234408908,
        "image_shape": [800, 1280]
    },
    "2": {
        "date": "2021-11-10 21:55:50",
        "description": "",
        "K": [[1164.80, 0.0, 660.27],
            [0.0, 1154.15, 405.47],
            [ 0.0, 0.0, 1.0]],
        "K_new": [[1164.80, 0.0, 660.27],
            [0.0, 1154.15, 405.47],
            [ 0.0, 0.0, 1.0]],
        "dist": [ 0.00, 0, 0,0, 0,0, 0,0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0],
        "reproj_error": 0.22981242234408908,
        "image_shape": [800, 1280]
    },
    "3": {
        "date": "2021-11-10 21:55:50",
        "description": "",
        "K": [[1390.24, 0.0, 662.89],
            [0.0,1391.94,372.84],
            [ 0.0, 0.0, 1.0]],
        "K_new": [[1390.24, 0.0, 662.89],
            [0.0,1391.94,372.84],
            [ 0.0, 0.0, 1.0]],
        "dist": [ 0,0, 0,0, 0,0, 0,0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0],
        "reproj_error": 0.22981242234408908,
        "image_shape": [800, 1280]
    },
    "4": {
        "date": "2021-11-10 21:55:50",
        "description": "",
        "K": [[1154.45, 0.0, 653.99],
            [0.0,1149.59,410.80],
            [ 0.0, 0.0, 1.0]],
        "K_new": [[1154.45, 0.0, 653.99],
            [0.0,1149.59,410.80],
            [ 0.0, 0.0, 1.0]],
        "dist": [ 0,0, 0,0, 0,0, 0,0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0],
        "reproj_error": 0.22981242234408908,
        "image_shape": [800, 1280]
    },
    "5": {
        "date": "2021-11-10 21:55:50",
        "description": "",
        "K": [[1338.87, 0.0, 646.03],
            [0.0,1347.73,412.16],
            [ 0.0, 0.0, 1.0]],
        "K": [[1338.87, 0.0, 646.03],
            [0.0,1347.73,412.16],
            [ 0.0, 0.0, 1.0]],
        "dist": [ 0,0, 0,0, 0,0, 0,0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0],
        "reproj_error": 0.22981242234408908,
        "image_shape": [800, 1280]
    },
    "6": {
        "date": "2021-11-10 21:55:50",
        "description": "",
        "K": [[1368.82, 0.0, 632.80],
            [0.0,1355.44, 442.61],
            [ 0.0, 0.0, 1.0]],
        "K_new": [[1368.82, 0.0, 632.80],
            [0.0,1355.44, 442.61],
            [ 0.0, 0.0, 1.0]],
        "dist": [ 0,0, 0,0, 0,0, 0,0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0],
        "reproj_error": 0.22981242234408908,
        "image_shape": [800, 1280]
    },
    "7": {
        "date": "2021-11-10 21:55:50",
        "description": "",
        "K": [[1150, 0.0, 640],
            [0.0,1150,400],
            [ 0.0, 0.0, 1.0]],
        "K_new": [[1150, 0.0, 640],
            [0.0,1150,400],
            [ 0.0, 0.0, 1.0]],
        "dist": [ 0,0, 0,0, 0,0, 0,0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0],
        "reproj_error": 0.22981242234408908,
        "image_shape": [800, 1280]
    },
    "8": {
        "date": "2021-11-10 21:55:50",
        "description": "",
        "K": [[1123.77, 0.0, 646.29],
            [0.0,1118.91,432.92],
            [ 0.0, 0.0, 1.0]],
        "K_new": [[1123.77, 0.0, 646.29],
            [0.0,1118.91,432.92],
            [ 0.0, 0.0, 1.0]],
        "dist": [ 0,0, 0,0, 0,0, 0,0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0],
        "reproj_error": 0.22981242234408908,
        "image_shape": [800, 1280]
    },
    "9": {
        "date": "2021-11-10 21:55:50",
        "description": "",
        "K": [[1126.68, 0.0, 620.0],
            [0.0,1125.55,415.07],
            [ 0.0, 0.0, 1.0]],
        "K_new": [[1126.68, 0.0, 620.0],
            [0.0,1125.55,415.07],
            [ 0.0, 0.0, 1.0]],
        "dist": [ 0,0, 0,0, 0,0, 0,0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0],
        "reproj_error": 0.22981242234408908,
        "image_shape": [800, 1280]
    }
}
"""

intrinsics_rational_json1280x800x4 = """
{
    "0": {
        "date": "2022-9-20 21:55:50",
        "description": "",
        "K": [[873, 0.0, 597],
            [0.0, 870, 381],
            [ 0.0, 0.0, 1.0]],
        "K_new": [[873, 0.0, 597],
            [0.0, 870, 381],
            [ 0.0, 0.0, 1.0]],
        "dist": [-0.3334, 0.1779, 0, 0, 0.0],
        "reproj_error": 0.22981242234408908,
        "image_shape": [800, 1280]
    },
    "1": {
        "date": "2022-9-20 21:55:50",
        "description": "",
        "K": [[758, 0.0, 641.24],
            [0.0,756, 434],
            [ 0.0, 0.0, 1.0]],
        "K_new": [[758, 0.0, 641.24],
            [0.0,756, 434],
            [ 0.0, 0.0, 1.0]],
        "dist": [-0.3336,0.1178, 0, 0, 0.0],
        "reproj_error": 0.22981242234408908,
        "image_shape": [800, 1280]
    },
    "2": {
        "date": "2022-9-20 21:55:50",
        "description": "",
        "K": [[758, 0.0, 641.24],
            [0.0,756, 434],
            [ 0.0, 0.0, 1.0]],
        "K_new": [[758, 0.0, 641.24],
            [0.0,756, 434],
            [ 0.0, 0.0, 1.0]],
        "dist": [-0.3336,0.1178, 0, 0, 0.0],
        "reproj_error": 0.22981242234408908,
        "image_shape": [800, 1280]
    },
    "3": {
        "date": "2022-9-20 21:55:50",
        "description": "",
        "K": [[873, 0.0, 597],
            [0.0, 870, 381],
            [ 0.0, 0.0, 1.0]],
        "K_new": [[873, 0.0, 597],
            [0.0, 870, 381],
            [ 0.0, 0.0, 1.0]],
        "dist": [-0.3334, 0.1779, 0, 0, 0.0],
        "reproj_error": 0.22981242234408908,
        "image_shape": [800, 1280]
    }
}
"""



intrinsics_rational_json800x600x6 = """
{
    "0": {
        "date": "2021-11-10 21:55:50",
        "description": "",
        "K": [[1150, 0.0, 400],
            [0.0,1150,300],
            [ 0.0, 0.0, 1.0]],
        "dist": [ 0,0, 0,0, 0,0, 0,0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0],
        "reproj_error": 0.1,
        "image_shape": [600, 800]
    },
    "1": {
        "date": "2021-11-10 21:55:50",
        "description": "",
        "K": [[1150, 0.0, 400],
            [0.0,1150,300],
            [ 0.0, 0.0, 1.0]],
        "dist": [ 0,0, 0,0, 0,0, 0,0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0],
        "reproj_error": 0.1,
        "image_shape": [600, 800]
    },
    "2": {
        "date": "2021-11-10 21:55:50",
        "description": "",
        "K": [[1150, 0.0, 400],
            [0.0,1150,300],
            [ 0.0, 0.0, 1.0]],
        "dist": [ 0,0, 0,0, 0,0, 0,0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0],
        "reproj_error": 0.1,
        "image_shape": [600, 800]
    },
    "3": {
        "date": "2021-11-10 21:55:50",
        "description": "",
        "K": [[1150, 0.0, 400],
            [0.0,1150,300],
            [ 0.0, 0.0, 1.0]],
        "dist": [ 0,0, 0,0, 0,0, 0,0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0],
        "reproj_error": 0.1,
        "image_shape": [600, 800]
    },
    "4": {
        "date": "2021-11-10 21:55:50",
        "description": "",
        "K": [[1150, 0.0, 400],
            [0.0,1150,300],
            [ 0.0, 0.0, 1.0]],
        "dist": [ 0,0, 0,0, 0,0, 0,0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0],
        "reproj_error": 0.1,
        "image_shape": [600, 800]
    },
    "5": {
        "date": "2021-11-10 21:55:50",
        "description": "",
        "K": [[1150, 0.0, 400],
            [0.0,1150,300],
            [ 0.0, 0.0, 1.0]],
        "dist": [ 0,0, 0,0, 0,0, 0,0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0],
        "reproj_error": 0.1,
        "image_shape": [600, 800]
    }
}
"""