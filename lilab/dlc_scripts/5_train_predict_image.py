##
'''
conda deactivate
module load anaconda3/4.7.12
source activate dlc2.2i  DLC-GPU
module load cuda/10.0 cudnn/7.6.4 ffmpeg/4.2.1 ##unable from 0826
export DLClight=True
'''

#Open IPython
#ipython
#import DeepLabCut (Step 1) 
import tensorflow as tf
import deeplabcut,os
from deeplabcut.refine_training_dataset import tracklets
###### note for each running
##os.environ['CUDA_VISIBLE_DEVICES'] = '-1'##CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 
## add GPU 
sess = tf.Session()

##
project='/home/liying_lab/taoxianming/DATA/rat/data/deeplabcut/sideviews7_dlc_640x480'
##
videos=os.path.join(project,'videos/640_480.avi')
##
config_path=os.path.join(project,'config.yaml')##640 x 480

###Extract frames (Step 4) 
#deeplabcut.extract_frames(config_path)

###Label frames (Steps 5 and 6)###need gui (windows)
#deeplabcut.label_frames(config_path)

###Check labels (optional)(Step 7) 
#deeplabcut.check_labels(config_path)


###Create training dataset (Step 8) 
##!!!!!!use cropped images only!!!!!!!! in labeled_data/..._croped
cropped_path=os.path.join(project,'labeled-data/640_480_cropped')
#os.mkdir(cropped_path)
deeplabcut.cropimagesandlabels(config_path,excludealreadycropped=False) ##early version--DLC-GPU, latest on GUI
##or copy files from manual_label_path(../labeled-data/2021-05-24_16-29-39) to cropped_path(../labeled-data/2021-05-24_16-29-39_cropped) !!!!!!!

##
deeplabcut.create_training_dataset(config_path,windows2linux=True,augmenter_type='imgaug')

###Train the network (Step 9) 
## maxiters=500000 edited by multi_step (the end line) in  dlc-models/iteration-0/.../pose_cfg.yaml
#batch_size=10 #dlc-models/iteration-0/.../pose_cfg.yaml
deeplabcut.train_network(config_path,displayiters=1000,saveiters=10000)

###Evaluate the trained network (Step 11) 
#eva_stat=deeplabcut.evaluate_network(config_path)
deeplabcut.evaluate_network(config_path,plotting=True)


##predict multi-position
video_preds=videos
pathRes=os.path.join(project,'videos')
###prediction
deeplabcut.analyze_videos(config_path, [video_preds], videotype='avi', gputouse=1,save_as_csv=True,batchsize=20,destfolder=pathRes)
#
#
###all detections video
#deeplabcut.create_video_with_all_detections(config_path, [video_preds], videotype='mp4')##latest version, 2.2
deeplabcut.create_video_with_all_detections(config_path, [video_preds],DLCscorername='DLC_resnet50_6camera-90fpsMay28shuffle1_230000')##later version, 2.2

##predict multi-position
pathRes=os.path.join(project,'pred_videos')
video_preds=[os.path.join(pathRes,i) for i in os.listdir(pathRes)]
###prediction
deeplabcut.analyze_videos(config_path, video_preds, videotype='avi', gputouse=1,save_as_csv=True,batchsize=20,destfolder=pathRes)
#
#
###all detections video
#deeplabcut.create_video_with_all_detections(config_path, [video_preds], videotype='mp4')##latest version, 2.2
deeplabcut.create_video_with_all_detections(config_path, video_preds,DLCscorername='DLC_resnet50_6camera-90fpsMay28shuffle1_400000')##later version, 2.2


deeplabcut.convert_detections2tracklets(config_path, [video_preds], videotype='mp4',track_method='ellipse',identity_only=True,overwrite = True)

  #
  ###detection to tracking
tm='ellipse'#box
#tmn='el'
deeplabcut.convert_detections2tracklets(config_path, [video_preds], videotype='mp4',track_method=tm,identity_only=False,overwrite = True)
#seg=videoName.replace('.mp4','')+'DLC_resnet50_6camera-90fpsMay28shuffle1_420000_'+tmn+'.pickle'
#tracks_pickle=os.path.join(project,'videos',seg)
##
deeplabcut.stitch_tracklets(
      config_path,
      [video_preds],
      "mp4",
      output_name='',
  )
##
print("Plotting trajectories...")
deeplabcut.plot_trajectories(
  config_path, [video_preds], "mp4", track_method=tm
)
##
print("Creating labeled video...")
deeplabcut.create_labeled_video(
  config_path,
  [video_preds],
  "mp4",
  save_frames=False,
  color_by="individual",
  track_method="ellipse",
)
print("Labeled video created.")
print("Filtering predictions...")
deeplabcut.filterpredictions(
  config_path, [video_preds], "mp4", track_method="ellipse"
)
deeplabcut.create_labeled_video(
  config_path,
  [video_preds],
  "mp4",
  save_frames=False,
  filtered=True,
  color_by="individual",
  track_method=tm,
  draw_skeleton=True,
)
print("Predictions filtered.")



deeplabcut.filterpredictions(config_path,[video_preds], videotype='mp4',track_method='box',filtertype='median')
