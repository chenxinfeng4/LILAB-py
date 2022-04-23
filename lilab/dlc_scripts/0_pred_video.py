import deeplabcut
import glob
# dlc no seg
config_path = '/home/liying_lab/chenxinfeng/deeplabcut-project/bwrat_28kpt-cxf-2022-02-25/config.yaml'
video_path = '/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_30fps/whiteblack/15-37-42-white-copy/'
video_preds = glob.glob(video_path + '*output_?.mp4')
deeplabcut.analyze_videos(config_path, video_preds, videotype='avi', gputouse=1,save_as_csv=True,batchsize=20,destfolder=video_path)


# dlc seg
config_path = '/home/liying_lab/chenxinfeng/deeplabcut-project/abc-edf-2021-11-23/config.yaml'
video_path = '/home/liying_lab/chenxinfeng/DATA/multiview-project/2021-11-02-bwrat_800x600_30fps/whiteblack/15-37-42-white-copy/dlc2d_seg/'
video_preds = glob.glob(video_path + '*output_?_white.avi') + glob.glob(video_path + '*output_?_black.avi')
deeplabcut.analyze_videos(config_path, video_preds, videotype='avi', gputouse=1,save_as_csv=True,batchsize=40,destfolder=video_path)