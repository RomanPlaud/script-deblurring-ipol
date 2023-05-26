import multiprocessing
from utils_unet.utils_unet import downsampling
import cv2

vcapture = cv2.VideoCapture(args.video_path)
length = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))

width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
dim = (width, height)

size_img = (192, 192)

imgs = []
for i in range(2):
    img, success = vcapture.read()
    imgs.append(img)

pool = multiprocessing.Pool(processes=2)

results = pool.starmap(downsampling, zip(imgs, repeat(size_img)))

pool.close()
pool.join()