#!/bin/bash

NO_ALBUMENTATIONS_UPDATE=1 nohup python app.py --mode onnx &> console.out &
