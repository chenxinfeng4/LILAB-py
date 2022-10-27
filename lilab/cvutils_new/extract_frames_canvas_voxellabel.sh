vdir='/mnt/ftp.rat/multiview_9/bigger_rat'
ls $vdir/*.mp4 | xargs -n 1 -P 8 python -m lilab.cvutils_new.extract_frames_canvas

calibpkl='/mnt/ftp.rat/multiview_9/bigger_rat/ball/2022-10-11_17-19-12ball.calibpkl'
python -m lilab.dannce.s1_ball2mat $calibpkl
cp $calibpkl.mat $vdir/outframes
