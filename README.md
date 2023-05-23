# Face blurring 

# Overview 

This source code provides a PyTorch implementation of a Unet-like model that directly blurr faces. This part of the code relies of [DeOldify implementation](http://www.ipol.im/pub/art/2022/403/). A Yolo model implementation based on [this repo](https://github.com/elyha7/yoloface) can also be used to blurr faces.


## Usage

### Clone the repo
```
git clone https://github.com/RomanPlaud/script-deblurring-ipol.git
```

### Install requirements
```
pip install -r requirements.txt
```
### Download pretrained models
```
bash weights/./download_unet.sh 
```
### perform face blurring 
Images to blurr should be in a folder 

#### Using Unet method`

We can run directly but specifications are detailed below\

```
python3 main.py
```

With detailed arguments :\ 
```
python3 main.py \
    --images_folder <path to the folder> \
    --output_folder <path to store results> \
    --device 'cuda' \
    --size_img [512, 512]\
    --method 'unet'
```

-``images_folder`` the folder in which are stored images in which you want to blurr faces. We provide a set a 16 images stored in ``"original_images/"`` (which default folder)
-``device`` can be set to ``'cpu'`` if no GPU is available\
-``size_img`` correspond to height and width.. By default it is set to ``[512, 512]`` and it is fitted to blurr faces that are not in foreground. If faces are in the foreground you can set ``size_img = [192,192]``.\
In any case, it can be changed at your conveinance (but the more the input image is of high resolution the more it will take time to perform inference).

#### Using Yolo method

```
python3 main.py \
    --images_folder <path to the folder> \
    --output_folder <path to store results> \
    --device 'cuda' \
    --method 'yolo' \
    --target_size <target size>\
    --min_face <min face>\
```
-``device`` can be set to ``'cpu'`` if no GPU is available\
-``target_size`` correspond to height (of a square image) of the size of which we want to rescale input image. By default it is set to ``None`` that means the image is not rescaled to perform inference.\
-``min_face`` correspond to the mininmum size of a face we want to blurr (by default ``min_face = 0``)\