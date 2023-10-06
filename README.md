# ML_DENSE_OBJECT

## Note 
Please note that the main part of the code has been released, though we are still testing it to fix possible glitches. Thank you.

This implementation is built on top of torchvision Retinanet. The SKU110K dataset is provided in csv format compatible with the code CSV parser.
Dependencies include: ```pytorch```, ```torchvision```, ```tqdm```, and was tested  using ```Python 3.8``` and ```OpenCV```

## Dataset

<img src="https://raw.githubusercontent.com/eg4000/SKU110K_CVPR19/master/figures/benchmarks_comparison.jpg" width="750">

We compare between key properties for related benchmarks. **#Img.**: Number of images. **#Obj./img.**: Average items per image. **#Cls.**: Number of object classes (more implies a harder detection problem due to greater appearance variations). **#Cls./img.**: Average classes per image. **Dense**: Are objects typically densely packed together, raising potential overlapping detection problems?. **Idnt**: Do images contain multiple identical objects or hard to separate object sub-regions?. **BB**: Bounding box labels available for measuring detection accuracy?.

The dataset is provided for the exclusive use by the recipient and solely for academic and non-commercial purposes. 

The dataset can be downloaded from <a href="http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"> here</a> or <a href="https://drive.google.com/file/d/1iq93lCdhaPUN0fWbLieMtzfB1850pKwd">here</a>.

## Usage

Download the dataset and extract to directory data:
Create the folder classes in path"data/SKU110K/classes" Download the classes file from <a href="https://drive.google.com/file/d/1hZS3oFVFx-W64cMi4A7MrB4Fx124-Mcg/view?usp=sharing">here</a> 
example path: 
* data/SKU110K/annotations
* data/SKU110K/images
* data/SKU110K/classes/class_mappings.csv

1. Move to the directory: ```cd ML_DENSE_OBJECT```
2. Train ``` python -m src.training ```

## Output 
### Expected 
![alt text](https://raw.githubusercontent.com/DamNT055/ML_DENSE_OBJECT/main/pictures/expected.png "Logo Title Text 1")

### Predicted
![alt text](https://raw.githubusercontent.com/DamNT055/ML_DENSE_OBJECT/main/pictures/output.png "Logo Title Text 1")
