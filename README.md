## Install package
```
cd ML_DENSE_OBJECT
pip install -r requirements.txt
```

## Download data
```
mkdir -p src/dataraw
cd src/dataraw
gdown --id 1lCbqgY8M-0KDyk_aJAqVJPupZvZJbout
unzip sku.zip
cd ..

```

## Training
```
python training.py
```