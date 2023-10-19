## Install package

```
cd ML_DENSE_OBJECT
pip install -r requirements.txt
```

## Download data

```
mkdir -p src/dataraw
cd src/dataraw
gdown --id 1mTmlQ7qutDBuxz_JDH2hZ7y8JVRnSl1
unzip SKU_1.zip
cd ..
!gdown --id 1XvDCuTBCHrtqoFBm-vt2NcAB9eU9o3nX
unzip results.zip

```

## Training

```
python training.py
```
