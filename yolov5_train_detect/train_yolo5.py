
# pip install pyyaml
import yaml
import os
import glob
# pip install ipython
from IPython.display import Image, display

config = {
    "path": "/inclusions/dataset_yolo5",
    "train": "/inclusions/dataset_yolo5/train",
    "val": "/inclusions/dataset_yolo5/valid",
    "nc": 7,
    "names": ["hiden_trash", "metal_trash", "partlyhiden_trash", "rag_trash", "red_trash", "rope_trash", "shugar_trash"]
}

with open("../data.yaml", "w") as file:
    yaml.dump(config, file, default_flow_style=False)

SIZE = 898
BATCH_SIZE = 32
EPOCHS = 300
MODEL = "yolov5s"
WORKERS = 1
PROJECT = "inclusions_v"
RUN_NAME = f"{MODEL}_size{SIZE}_epochs{EPOCHS}_batch{BATCH_SIZE}_small"

# python train.py --img {SIZE}\
#                 --batch {BATCH_SIZE}\
#                 --epochs {EPOCHS}\
#                 --data ../inclusions/data.yaml\
#                 --weights {MODEL}.pt\
#                 --workers {WORKERS}\
#                 --project {PROJECT}\
#                 --name {RUN_NAME}\

runs_directory = "inclusions/dataset_yolo5/test/images"

display(
    Image(
        filename=f"{runs_directory}/IMG20221013124300_jpg.rf.82635337caf82fa42f9f757b31319be2.jpg"
    )
)