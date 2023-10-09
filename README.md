## Install package

```
cd ML_DENSE_OBJECT
pip install -r requirements.txt
```

## Download data

```
mkdir -p src/dataraw
cd src/dataraw
gdown --id 1baFn5FBNqNH3itjduq0msZ7TYn-yBXyz
unzip sku.zip
cd ..
!gdown --id 1XvDCuTBCHrtqoFBm-vt2NcAB9eU9o3nX
unzip results.zip

```

## Training

```
python training.py
```
