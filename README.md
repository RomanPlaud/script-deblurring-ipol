# script-deblurring-ipol

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
bash weights/./download_weights.sh

```
perform face blurring 
```
python main.py 
```
