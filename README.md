## Install package

```
cd ML_DENSE_OBJECT
pip install -r requirements.txt
```

## Download data

```
mkdir -p src/dataraw
cd src/dataraw
gdown --id 1--OrtW5jOmqpftWRiN956Jav_PE_nDiY
unrar x SKU_1.rar
cd ..
!gdown --id 1XvDCuTBCHrtqoFBm-vt2NcAB9eU9o3nX
unzip results.zip

```

## Training

```
python training.py
```
