#!/bin/bash
# Download latest models from https://github.com/ultralytics/yolov5/releases
# Usage:
#    $ bash weights/download_weights.sh

python3 - <<EOF
from utils_yolo.google_utils import attempt_download

for x in ['s', 'm']:
    attempt_download(f'yolov5{x}.pt')

EOF

mv yolov5s.pt weights/
mv yolov5m.pt weights/
