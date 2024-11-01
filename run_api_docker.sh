#!/bin/bash

NAME="livep-api"

sudo docker rm -f "$NAME" &> /dev/null
docker run -it \
    --volume=.:/root/FasterLivePortrait \
    --restart=always \
    --runtime=nvidia \
    --gpus="device=0" \
    --network=host \
    --name="$NAME" \
    binarii/faster_liveportrait:v3 \
    /bin/bash
