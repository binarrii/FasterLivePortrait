#!/bin/bash

sudo docker rm -f faster_liveportrait &> /dev/null
sudo docker run -d \
    --runtime=nvidia \
    --gpus=all \
    --name=faster_liveportrait \
    --restart=always \
    -v .:/root/FasterLivePortrait \
    -w /root/FasterLivePortrait \
    -p 9870:9870 \
    binarii/faster_liveportrait \
    bash webui.sh

