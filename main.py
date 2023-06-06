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
import datetime
from tqdm import trange
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser(description='Face blurring')

    parser.add_argument('--images_folder', help='path to the images folder', default="example_images/")
    parser.add_argument('--output_folder', help='path to the output folder', default="output/")
    parser.add_argument('--path_model', help='path to the model', default="weights/deblurring_model_mse.pth")
    parser.add_argument('--device', help='device to use', default="cuda")
    parser.add_argument('--size_img', help='size of the image', default=[512,512], type=int, nargs='+')
    parser.add_argument('--method', help='method to use', default="unet")

    ## YOLO parameters
    parser.add_argument('--target_size', help='target size of the image', default=None, type=int)
    parser.add_argument('--min_face', help='minimum size of the face', default=0, type=int)

    ## Video parameters
    parser.add_argument('--mode_video', help='if you want to perform inference on a video', default=False)
    parser.add_argument('--video_path', help='path to the video', default='example_videos/vid_example.mp4')
    parser.add_argument('--video_output', help='path to the output video', default="output_videos/")


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if not args.mode_video:
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
                img = inference(img, model, tuple(args.size_img), args.device)
                img.save(path_save)

        elif args.method == "yolo":

            ## LOAD NETWORK
            model = YoloDetector(target_size=args.target_size, device=args.device, min_face=args.min_face)
            
            ## INFERENCE
            for path in os.listdir(args.images_folder):
                img = Image.open(os.path.join(args.images_folder, path))
                path_save = os.path.join(args.output_folder, path)
                img = inference_yolo(img, model)
                img.save(path_save)

        else : 
            raise("Error : Method not implemented")
        
    else:

        ## create output folder 
        if not os.path.exists(args.video_output):
            os.makedirs(args.video_output)

        vcapture = cv2.VideoCapture(args.video_path)
        length = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))

        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dim = (width, height)
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(args.video_output, f"video_blurred_{current_time}.avi")

        vwriter = cv2.VideoWriter(file_name,
                                    cv2.VideoWriter_fourcc('F','M','P','4'),
                                    fps, dim)

        if args.method == "unet":
            ## LOAD NETWORK
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            y_range=torch.Tensor([-3.,-3.,-3.]), torch.Tensor([3.,3.,3.])
            path_model = args.path_model
            model = Unet(n_channels=3, pretrained=path_model, backbone=backbone, y_range=y_range, spectral=True)

            ## INFERENCE
            times = []
            for _ in trange(length):
                success, img = vcapture.read()
                if not success : 
                    break
                img = Image.fromarray(img[:,:,[2,1,0]])
                output = inference(img, model, size_img=tuple(args.size_img), device=args.device)
                output = np.array(output)
                vwriter.write(output[:,:,[2,1,0]])

            vwriter.release()
            times = times[2:]
            print(np.mean(times), np.std(times), len(times))
            print(1/np.mean(times))
        
        elif args.method == "yolo":

            ## LOAD NETWORK
            model = YoloDetector(target_size=args.target_size, device=args.device, min_face=args.min_face)
            
            ## INFERENCE
            for _ in trange(length):
                success, img = vcapture.read()
                if not success : 
                    break
                img = Image.fromarray(img[:,:,[2,1,0]])
                output = inference_yolo(img, model)
                output = np.array(output)
                vwriter.write(output[:,:,[2,1,0]])
                
            vwriter.release()

        else : 
            raise("Error : Method not implemented")

        
