import cv2
from utils_yolo.face_detector import YoloDetector
import numpy as np
from PIL import Image
import os

def blurr_bboxes(image, bboxes, factor=1):

    tempImg = image.copy()
    w, h = image.shape[:2]
    mask = np.full((w,h,1), 0, dtype=np.uint8)

    k = 0
    for (x, y, u, v) in bboxes:
        k = max(k, max(abs(x-u), abs(y-v)))

        center = ((x+u)//2, (y+v)//2)
        h = abs(x-u)
        w = abs(y-v)
        cv2.ellipse(mask, center, (h//2, w//2), 0, 0, 360, 255, -1)

    k = int(k)//factor
    k = (k%2==0)*(k+1) + (k%2==1)*k
    tempImg = cv2.GaussianBlur(tempImg, (k,k), 0)


    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(image,image,mask = mask_inv)
    img2_fg = cv2.bitwise_and(tempImg,tempImg,mask = mask)
    dst = cv2.add(img1_bg,img2_fg)

    return dst

def inference_yolo(img, model):

    bboxes, _ = model.predict(np.array(img))
    
    output = blurr_bboxes(np.array(img), bboxes[0])
    res = Image.fromarray(output)

    return res