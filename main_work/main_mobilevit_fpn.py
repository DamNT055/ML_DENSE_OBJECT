import os
import sys  

import csv 
import cv2
import torch 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torchvision.io import read_image
from six import raise_from
from tqdm import tqdm

from get_image_size import get_image_size
from engine import train_one_epoch, evaluate

import utils_vision 
import transforms as T 
import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

sys.path.insert(0, "d:\\CAPSTONE\\mlcvnets")
from data import create_train_val_loader
from options.utils import load_config_file
from options.opts import get_training_arguments

from cvnets import get_model
from cvnets.models.detection import build_detection_model
from cvnets.models.classification import arguments_classification, build_classification_model
from torchvision.models.detection import RetinaNet

os.chdir("D:/CAPSTONE/mlcvnets")
print(os.getcwd())


from torch import nn
from typing import Callable, Dict, List, Optional, Union
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool, LastLevelP6P7
from torchvision.models.detection.backbone_utils import BackboneWithFPN

def _mobilevit_fpn_extractor(
    backbone,
    trainable_layers: int,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
):    
    layers_to_train = ["conv_1x1_exp", "layer_5", "layer_4", "layer_3", "layer_2", "layer_1", "conv_1"]
    for name, parameter in backbone.named_modules():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)
    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()
    if returned_layers is None: 
        returned_layers = [1,2,3,4,5]
    if min(returned_layers)<=0 or max(returned_layers)>=6:
        raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")
    returned_layers = {f"layer_{k}": str(v) for v,k in enumerate(returned_layers)}
    in_channels_list = []
    for layer in layers_to_train[::-1]:
        if layer.split('_')[0] == 'layer':
            in_channels_list.append(getattr(backbone, layer)[0].out_channels)
        else: in_channels_list.append(getattr(backbone, layer).out_channels)
    out_channels=256
    print("in_channels_list: ", in_channels_list)
    print("====> Done !")
    return BackboneWithFPN(backbone=backbone, return_layers=returned_layers, in_channels_list=in_channels_list, out_channels=out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer)

"""
"""

def _parse(value, function, fmt):
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)
def _open_for_csv(path):
    return open(path, 'r', newline='')
def get_image_metadata(file_path):
    size = os.path.getsize(file_path)
def _read_classes(csv_reader):
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1
        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError(
                'line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(
            class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError(
                'line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
        return result
def _read_images(base_dir):
    result = {}
    #dirs = [os.path.join(base_dir, o) for o in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, o))]
    dirs = [os.path.join(base_dir, 'images')]
    if len(dirs) == 0:
        dirs = ['']
    for project in dirs:
        project_imgs = os.listdir(os.path.join(base_dir, project))
        i = 0
        print("Loading images...")
        for image in tqdm(project_imgs):
            try:
                img_file = os.path.join(base_dir, project, image)
                exists = os.path.isfile(img_file)
                
                if not exists:
                    print("Warning: Image file {} is not existing".format(img_file))
                    continue
                # Image shape
                height, width = get_image_size(img_file)
                result[img_file] = {"width": width, "height": height}
                i += 1
            except Exception as e:
                print("Error: {} in image: {}".format(str(e), img_file))
                continue
    return result
def _read_annotations(csv_reader, classes, base_dir, image_existence):
    result = {}
    base_dir = os.path.join(base_dir, 'images')
    for line, row in enumerate(csv_reader):
        line += 1
        try: 
            img_file, x1, y1, x2, y2, class_name, width, height = row[:]
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            width = int(width)
            height = int(height)
            if x1 >= width:
                x1 = width - 1
            if x2 >= width:
                x2 = width - 1
            
            if y1 >= height:
                y1 = height - 1
            if y2 >= height:
                y2 = height - 1 
            # x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0
            if x1<0 | y1<0 or x2<=0 or y2<=0:
                print("Warning: Image file {} has some bad boxes annotations".format(img_file))
                continue
            # Append root path 
            img_file = os.path.join(base_dir, img_file)
            # Check images exists 
            if img_file not in image_existence:
                print("Warning: Image file {} is not existing".format(img_file))
                continue                
        except ValueError:
            raise_from(ValueError(
                'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)),
                None)
        if img_file not in result:
            result[img_file] = []
        # If a row contains only an image path, it's an image without annotations.
        if (x1,x2,y1,y2,class_name) == ('', '', '', '', ''):
            continue
        x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))
        # Check that the bounding box is valid.
        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))
        # check if the current class name is correctly present
        if class_name not in classes:
            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))
        result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
    return result

"""
"""

class CSVGenerator(torch.utils.data.Dataset):
    def __init__(self, csv_data_file, csv_class_file, width, height, base_dir=None, transform = None, **kwargs):
        self.image_names = []
        self.image_data = {}
        self.base_dir = base_dir
        self.transform = transform
        self.height = height
        self.width = width

        # __ delete __
        self.csv_data_file = csv_data_file
        self.csv_class_file = csv_class_file
        # __ delete __

        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            self.base_dir = os.path.dirname(os.path.dirname(csv_data_file))

        # Parse the provided class file
        try:
            with _open_for_csv(csv_class_file) as file:
                self.classes = _read_classes(csv.reader(file, delimiter=','))
                # classes['object] = 0
        except ValueError as e:
            raise_from(ValueError(
                'invalid CSV class file: {}: {}'.format(csv_class_file, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key  # labels[1] = object
            
        # build mappings for existence
        self.image_existence = _read_images(self.base_dir)
        
        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with _open_for_csv(csv_data_file) as file:
                self.image_data = _read_annotations(csv.reader(file, delimiter=','), self.classes, self.base_dir, self.image_existence)
        except ValueError as e: 
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_data_file, e)), None)
        self.image_names = list(self.image_data.keys())
        
    def __getitem__(self, index):
        img_name = image_path = self.image_names[index]
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        img_res = img_res/255.0
        boxes = []
        labels = []
        wt = img.shape[1]
        ht = img.shape[0]
        for row in self.image_data.get(img_name):
            #{'x1': 208, 'x2': 422, 'y1': 537, 'y2': 814, 'class': 'object'}
            x1, x2, y1, y2, class_name = row['x1'], row['x2'], row['y1'], row['y2'], row['class']
            labels.append(self.classes.get(class_name))
            xmin_corr = (x1/wt)*self.width
            xmax_corr = (x2/wt)*self.width
            ymin_corr = (y1/ht)*self.height
            ymax_corr = (y2/ht)*self.height
            boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.as_tensor([index])
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        
        if self.transform is not None:
            sample = self.transform(image = img_res, bboxes = target['boxes'], labels = labels)
            img_res = sample['image']        
            target['boxes'] = torch.Tensor(sample['bboxes'])
        return img_res, target
       
    def __len__(self):
        return len(self.image_names)
        
"""
"""

# Send train=True fro training transforms and False for val/test transforms
def get_transform(train):

    if train:
        return A.Compose([
                            A.HorizontalFlip(0.5),
                     # ToTensorV2 converts image to pytorch tensor without div by 255
                            ToTensorV2(p=1.0)
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
                            ToTensorV2(p=1.0)
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

"""
"""        


"""
dataset = CSVGenerator(
    csv_data_file= os.path.abspath("../SKU110K_fixed/annotations/annotations_train.csv"),
    csv_class_file= os.path.abspath("../SKU110K_fixed/classes/class_mappings.csv"),
    width=224,
    height=224,
    transform=get_transform(train=True)
)
dataset_test = CSVGenerator(
    csv_data_file= os.path.abspath("../SKU110K_fixed/annotations/annotations_train.csv"),
    csv_class_file= os.path.abspath("../SKU110K_fixed/classes/class_mappings.csv"),
    width=224,
    height=224,
    transform=get_transform(train=False)
)

torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()

test_split = 0.2
tsize = int(len(dataset)*test_split)
dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-tsize:])

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=8, shuffle=True, #num_workers=0,
    collate_fn=utils_vision.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=4, shuffle=False, #num_workers=0,
    collate_fn=utils_vision.collate_fn)
"""


"""
"""

opts = get_training_arguments()
setattr(opts, "common.config_file", "config/detection/ssd_mobilevitv3_xx_small_320.yaml")
opts = load_config_file(opts=opts)

setattr(opts, "model.detection.n_classes", 81)
setattr(opts, "dataset.workers", 0)

for key, val in vars(opts).items():
    print(key, ":", val)

"""
"""
mobilevit = build_classification_model(opts=opts)
print("mobilevit !")
backbone = _mobilevit_fpn_extractor(backbone=mobilevit, trainable_layers=1, extra_blocks=LastLevelP6P7(256, 256))
print("mobilevit with fpn !")
exit()

num_classes = 2
model = RetinaNet(backbone=backbone, num_classes=num_classes)

"""
"""

images, targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k:v for k,v in t.items()} for t in targets]
output = model(images, targets)

print(output)

print("done !")