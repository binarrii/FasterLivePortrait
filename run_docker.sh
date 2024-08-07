#!/bin/bash

NAME="faster_liveportrait"

sudo docker rm -f "$NAME" &> /dev/null
sudo docker run \
    -it \
    --runtime=nvidia \
    --gpus=all \
    --restart=always \
    --name="$NAME" \
    -v .:/root/FasterLivePortrait \
    -w /root/FasterLivePortrait \
    -p 9870:9870 \
    binarii/faster_liveportrait \
    /bin/bash
