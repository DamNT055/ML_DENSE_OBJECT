import sys 
import os 
sys.path.insert(0, "d:\\CAPSTONE\\mlcvnets")

import torch 
from cvnets import get_model
from torchvision.io import read_image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.patches as patches

from options.utils import load_config_file
from options.opts import get_training_arguments
from cvnets.models.detection import build_detection_model
from data import create_train_val_loader

opts = get_training_arguments()

setattr(opts, "common.config_file", "config/detection/ssd_mobilevitv3_xx_small_320.yaml")
opts = load_config_file(opts=opts)
print(opts)

setattr(opts, "model.detection.n_classes", 2)
setattr(opts, "dataset.workers", 0)

train_loader, val_loader, train_sampler = create_train_val_loader(opts)
next(iter(train_loader))