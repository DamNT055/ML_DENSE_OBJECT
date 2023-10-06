import os
import keras
import tarfile

url = "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz"
filename = os.path.join(os.getcwd(), "data.tar.gz")
keras.utils.get_file(filename, url)

with tarfile.open("data.tar.gz", "r") as z_fp:
    z_fp.extractall("../data")

os.makedirs("../data/SKU110_fixed/classes")

url = "https://drive.google.com/file/d/1hZS3oFVFx-W64cMi4A7MrB4Fx124-Mcg/view?usp=sharing"
filename = os.path.join(os.getcwd(), "class_mappings.csv")
keras.utils.get_file(filename, url)