import os 
import csv
import cv2
import torch
import numpy as np
from six import raise_from
from preprocessing import _open_for_csv, _read_classes, _read_images, _read_annotations
from transform import get_transform
from vision_utils.utils_vision import collate_fn

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

def dataloader_generator():
    dataset = CSVGenerator(
        csv_data_file= os.path.abspath("dataraw/SKU110K/annotations/annotations_train.csv"),
        csv_class_file= os.path.abspath("dataraw/SKU110K/classes/class_mappings.csv"),
        width=2048,
        height=2048,
        transform=get_transform(train=True)
    )
    dataset_test = CSVGenerator(
        csv_data_file= os.path.abspath("dataraw/SKU110K/annotations/annotations_val.csv"),
        csv_class_file= os.path.abspath("dataraw/SKU110K/classes/class_mappings.csv"),
        width=2048,
        height=2048,
        transform=get_transform(train=False)
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=True, num_workers=0,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=5, shuffle=False, num_workers=0,
        collate_fn=collate_fn)
    
    return data_loader, data_loader_test