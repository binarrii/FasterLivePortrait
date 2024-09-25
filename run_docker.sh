#!/bin/bash

NAME="livep"

sudo docker rm -f "$NAME" &> /dev/null
docker run -d \
    -p 9090:9090 \
    --volume=./ph:/root/FasterLivePortrait/ph \
    --volume=./pc:/root/FasterLivePortrait/pc \
    --restart=always \
    --runtime=nvidia \
    --gpus="device=0" \
    --name="$NAME" \
    binarii/livep:202409231300
