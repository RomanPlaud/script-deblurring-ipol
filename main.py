import argparse
import cv2
import torch
from torchvision import models
from utils_unet.unet import Unet
from utils_unet.utils_unet import inference
from utils_yolo.utils_yolo import inference_yolo
import os
from PIL import Image
from utils_yolo.face_detector import YoloDetector



def parse_args():
    parser = argparse.ArgumentParser(description='Face Blurring')

    parser.add_argument('--images_folder', help='path to the images folder', default="images/")
    parser.add_argument('--output_folder', help='path to the output folder', default="output/")
    parser.add_argument('--path_model', help='path to the model', default="weights/deblurring_model.pth")
    parser.add_argument('--device', help='device to use', default="cuda")
    parser.add_argument('--size_img', help='size of the image', default=[192,192], type=int, nargs='+')
    parser.add_argument('--method', help='method to use', default="unet")

    ## YOLO parameters
    parser.add_argument('--target_size', help='target size of the image', default=None, type=int)
    parser.add_argument('--min_face', help='minimum size of the face', default=0, type=int)


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    ## create output folder 
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    if args.method == "unet":
        ## LOAD NETWORK
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        y_range=torch.Tensor([-3.,-3.,-3.]), torch.Tensor([3.,3.,3.])
        path_model = args.path_model
        model = Unet(n_channels=3, pretrained=path_model, backbone=backbone, y_range=y_range, spectral=True)

        ## INFERENCE
        for path in os.listdir(args.images_folder):
            img = Image.open(os.path.join(args.images_folder, path))
            path_save = os.path.join(args.output_folder, path)
            inference(img, model, path_save, tuple(args.size_img), args.device)
    
    elif args.method == "yolo":

        ## LOAD NETWORK
        model = YoloDetector(target_size=args.target_size, device=args.device, min_face=args.min_face)
        
        ## INFERENCE
        for path in os.listdir(args.images_folder):
            img = Image.open(os.path.join(args.images_folder, path))
            path_save = os.path.join(args.output_folder, path)
            inference_yolo(img, model, path_save)

    else : 
        raise("Error : Method not implemented")
    

