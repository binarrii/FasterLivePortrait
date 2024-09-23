#!/bin/bash

NAME="faster_liveportrait_dev"

sudo docker rm -f "$NAME" &> /dev/null
sudo docker run \
    -it \
    --restart=always \
    --network=host \
    --runtime=nvidia \
    --gpus="device=0" \
    --name="$NAME" \
    -e NO_ALBUMENTATIONS_UPDATE=1 \
    -v .:/root/FasterLivePortrait \
    -w /root/FasterLivePortrait \
    binarii/faster_liveportrait:v3 \
    /bin/bash
