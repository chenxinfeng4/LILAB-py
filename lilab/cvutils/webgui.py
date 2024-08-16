# python -m lilab.cvutils.webgui
from pywebio.input import *
from pywebio.output import *
from pywebio import pin
from pywebio import start_server
import numpy as np
import os
import os.path as osp
import glob
import shutil
import cv2
import mmcv
import os
from subprocess import PIPE, run


def runargs_show(uimessage, args):
    # run the command, and get the stdout and stderr and returncode
    clear(uimessage)
    result = run(args, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    with use_scope(uimessage, clear=True):
        put_text(result.stdout)
        put_error(result.stderr)
        if result.returncode == 0:
            put_success("Success!")
        else:
            put_error("Fail!")


def on_extract_frames(uiinput_path, uimessage):
    clear(uimessage)
    input_path = pin.pin[uiinput_path]
    module = "lilab.cvutils.extract_frames"

    args = ["python", "-m", module, input_path]
    runargs_show(uimessage, args)


def on_crop_video(uifiletype, uiinput_path, uimessage):
    clear(uimessage)
    input_path = pin.pin[uiinput_path]
    if pin.pin[uifiletype] == "image":
        module = "lilab.cvutils.crop_image"
    else:
        module = "lilab.cvutils.crop_videofast"

    args = ["python", "-m", module, input_path]
    runargs_show(uimessage, args)


def on_concat_video(uifiletype, uiinput_path, uiinput_path2, uimessage):
    clear(uimessage)
    input_path = pin.pin[uiinput_path]
    input_path2 = pin.pin[uiinput_path2]
    if "image" in pin.pin[uifiletype]:
        module = "lilab.cvutils.concat_image"
    else:
        module = "lilab.cvutils.concat_video"

    args = ["python", "-m", module, input_path, input_path2]
    runargs_show(uimessage, args)


def on_concat_videopro(uiinput_textarea, uimessage):
    clear(uimessage)
    module = "lilab.cvutils.concat_videopro"
    videos = pin.pin[uiinput_textarea].split("\n")
    checks = [osp.exists(video) for video in videos]
    if not len(videos) or not all(checks):
        with use_scope(uimessage, clear=True):
            put_error("Not all videos exist!")
        return
    args = ["python", "-m", module] + videos
    runargs_show(uimessage, args)


def on_prepare_config_load(uiinput_path, ui_scopecode, uimessage):
    clear(uimessage)
    input_path = pin.pin[uiinput_path]
    config_path = osp.join(input_path, "config_video.py")
    if not osp.exists(config_path):
        with use_scope(uimessage, clear=True):
            put_error("No config.py found!")
        return
    else:
        with use_scope(uimessage, clear=True):
            put_success("Success!")
        with open(config_path, "r") as f, use_scope(ui_scopecode, clear=True):
            config = f.read()
            put_code(config, language="python")


def on_prepare_config_default(uiinput_path, ui_scopecode, uimessage):
    config_path = (
        "/home/liying_lab/chenxinfeng/ml-project/multiview_scripts/config_video.py"
    )
    with open(config_path, "r") as f, use_scope(ui_scopecode, clear=True):
        config = f.read()
        put_code(config, language="python")
    with use_scope(uimessage, clear=True):
        put_success("Success!")


def on_prepare_config_save(uiinput_path, ui_scopecode, uimessage):
    input_path = pin.pin[uiinput_path]
    config_path = osp.join(input_path, "config_video.py")
    default_config_path = (
        "/home/liying_lab/chenxinfeng/ml-project/multiview_scripts/config_video.py"
    )
    if osp.isdir(input_path):
        # copy without overwrite
        if not osp.isfile(config_path):
            shutil.copyfile(default_config_path, config_path)
            with use_scope(uimessage, clear=True):
                put_success("Copy Success!")
        else:
            with use_scope(uimessage, clear=True):
                put_error("File exists!")
    else:
        with use_scope(uimessage, clear=True):
            put_error("Not a directory!")


def app():
    put_tabs(
        [
            {"title": "prepare config.py", "content": put_scope("prepare_config")},
            {"title": "extract frames", "content": put_scope("extract_frames")},
            {"title": "crop image/video", "content": put_scope("crop_video")},
            {"title": "concat image/video", "content": put_scope("concat_video")},
            {"title": "concat seg video", "content": put_scope("concat_seg_video")},
        ]
    )
    with use_scope("prepare_config"):
        p_c_names = names = ["p_c_input_path", "p_c_scopecode", "p_c_message"]
        pin.put_input(names[0], label="The folder")
        put_button(
            "check&load exist", onclick=lambda: on_prepare_config_load(*p_c_names)
        )
        put_button(
            "load default", onclick=lambda: on_prepare_config_default(*p_c_names)
        )
        put_button("save", onclick=lambda: on_prepare_config_save(*p_c_names))
        put_scope(names[1])
        put_scope(names[2])
    with use_scope("extract_frames"):
        e_f_names = names = ["e_f_input_path", "e_f_message"]
        pin.put_input(names[0], label="The video path folder")
        put_button("Run", onclick=lambda: on_extract_frames(*e_f_names))
        put_scope(names[1])
    with use_scope("crop_video"):
        c_v_names = names = ["c_v_filetype", "c_v_input_path", "c_v_message"]
        pin.put_radio(names[0], options=["video", "image"], value="video", inline=True)
        pin.put_input(names[1], label="Input path folder")
        put_button("Run", onclick=lambda: on_crop_video(*c_v_names))
        put_scope(names[2])
    with use_scope("concat_video"):
        cc_names = names = [
            "cc_filetype",
            "cc_input_path1",
            "cc_input_path2",
            "cc_message",
        ]
        put_collapse(
            "Concat left and right videos/images",
            [
                pin.put_radio(
                    names[0],
                    options=["video file", "image folder"],
                    value="video file",
                    inline=True,
                ),
                pin.put_input(names[1], label="Input path [left]"),
                pin.put_input(names[2], label="Input path [right]"),
                put_button("Run", onclick=lambda: on_concat_video(*cc_names)),
            ],
            open=True,
        )
        put_collapse(
            "Concat multiple videos",
            [
                pin.put_textarea(
                    "cc_input_textarea",
                    label="Multiple videos",
                    rows=10,
                    code={"lineNumbers": True},
                ),
                put_button(
                    "Run",
                    onclick=lambda: on_concat_videopro(
                        "cc_input_textarea", cc_names[-1]
                    ),
                ),
            ],
            open=False,
        )
        put_scope(names[2])


if __name__ == "__main__":
    start_server(app, debug=True, port="44322")
