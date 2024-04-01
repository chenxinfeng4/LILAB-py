#!/bin/bash
#server
mkfifo ~/streampipe
nc -l 8081 >  ~/streampipe

# 先 puller
ffmpeg -f flv -i  ~/streampipe ~/out2.flv

# 后 pusher
ffmpeg -re -i out.mp4 -f flv ~/streampipe

