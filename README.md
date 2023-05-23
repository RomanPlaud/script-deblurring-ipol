# Face blurring 

# Overview 

This source code provides a PyTorch implementation of a Unet-like model that directly blurr faces. The code relies of [DeOldify implementattion](http://www.ipol.im/pub/art/2022/403/)

Clone the repo
```
git clone https://github.com/RomanPlaud/script-deblurring-ipol.git
```

Install requirements
```
pip install -r requirements.txt
```
Download pretrained models
```
bash weights/./download_unet.sh 
```
perform face blurring 
```
python main.py 
```
