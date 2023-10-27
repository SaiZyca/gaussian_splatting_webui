# gaussian_splatting_webui

for easy use gaussian_splatting

## colmap

This repo Contains only colmap webui,  
Follow these steps to install gaussian splatting in repositories folder

## gaussian splatting

* 設定 cmaker
```
  SET PATH=%PATH%; \your vc++ folder\VS2022\BuildTools\VC\Tools\MSVC\14.33.31629\bin\Hostx64\x64
```

```
git clone --recursive https://github.com/camenduru/gaussian-splatting

pip install -q plyfile

cd \gaussian-splatting\submodules\diff-gaussian-rasterization
python setup.py install

cd \gaussian-splatting\submodules\simple-knn
python setup.py install

wget https://huggingface.co/camenduru/gaussian-splatting/resolve/main/tandt_db.zip
unzip tandt_db.zip

python train.py -s /content/gaussian-splatting/tandt/train
```
