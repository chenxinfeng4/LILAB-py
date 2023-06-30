#create_labeled_video
import deeplabcut as dlc
import glob
import sys
import os

# set environment variable CUDA_VISIBLE_DEVICES to 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


print('1 {}\n2{}'.format(sys.argv[1], sys.argv[2]))

videos_folder=sys.argv[2]
config_name = sys.argv[1]

print('Processing:',videos_folder)

videos = glob.glob(videos_folder+'/*black.avi')
videos += glob.glob(videos_folder+'/*white.avi')
videos_labeled = glob.glob(videos_folder+'/*labeled.mp4')
videos_orginal = list(set(videos) - set(videos_labeled))
assert videos_orginal, 'No videos'
dlc.create_labeled_video(config_name, videos=videos_orginal)