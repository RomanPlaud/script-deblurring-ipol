import torch 
import numpy as np
from PIL import Image
import multiprocessing
from functools import partial
from itertools import repeat



def inference(img, model, size_img=(192, 192), device='cpu'):

    dataset_MEAN = torch.Tensor([0.485, 0.456, 0.406])
    dataset_STD = torch.Tensor([0.229, 0.224, 0.225])
    
    if next(model.parameters()).device != torch.device(device):
        model.to(device)

    model.eval()
    
    
    input_img_org = img.convert('RGB') 
    input_array = np.array(input_img_org)
    w, h, _ = input_array.shape
    
    input_img = input_img_org.resize(size_img, resample= Image.BILINEAR)
    input_img = torch.from_numpy(np.array(input_img,dtype=np.float32)).div_(255)
    input_img = input_img.sub_(other=dataset_MEAN).div_(other=dataset_STD)
    input_img = input_img.permute(2,0,1)[None]
    input_img = input_img.to(device)
                    
    output = model(input_img)[0]
            
    printer = output.detach().cpu().permute(1, 2, 0)
    printer = printer.mul(other=dataset_STD).add(other=dataset_MEAN)
    printer = printer.mul(255).numpy()
    result = Image.fromarray(printer.astype(np.uint8))

    upsample_output = result.resize((h,w), resample= Image.BILINEAR)
    upsample_output = np.array(upsample_output)

    mask = Image.fromarray((np.abs((output - input_img).squeeze().detach().cpu().numpy()).sum(axis=0))>0.1)
    mask = mask.resize((h,w), resample= Image.BILINEAR)
    mask_array = np.array(mask) 

    res = input_array.copy()
    mask_3d = np.repeat(mask_array[:, :, np.newaxis], 3, axis=2).astype(float)
    res = Image.fromarray((res * (1 - mask_3d) + upsample_output * mask_3d).astype(np.uint8))

    return res 

dataset_MEAN = torch.Tensor([0.485, 0.456, 0.406])
dataset_STD = torch.Tensor([0.229, 0.224, 0.225])

def downsampling(img, size_img=(192, 192)):

    
    input_img_org = img.convert('RGB') 
    input_array = np.array(input_img_org)
    w, h, _ = input_array.shape

    input_img = input_img_org.resize(size_img, resample= Image.BILINEAR)
    input_img = torch.from_numpy(np.array(input_img,dtype=np.float32)).div_(255)
    input_img = input_img.sub_(other=dataset_MEAN).div_(other=dataset_STD)

    input_img = input_img.permute(2,0,1)[None]


    return input_array, input_img, (w,h)

def forward_model(input_imgs, model, device='cpu'):

    if next(model.parameters()).device != torch.device(device):
        model.to(device)

    model.eval()

    input_imgs = input_imgs.to(device)
    outputs = model(input_imgs)

    outputs = outputs.detach().cpu()
    
    return outputs

def upsampling(input_array, input_img, output, dim):

    w, h = dim

    printer = output.detach().cpu().permute(1, 2, 0)
    printer = printer.mul(other=dataset_STD).add(other=dataset_MEAN)
    printer = printer.mul(255).numpy()
    result = Image.fromarray(printer.astype(np.uint8))

    upsample_output = result.resize((h,w), resample= Image.BILINEAR)
    upsample_output = np.array(upsample_output)

    mask = Image.fromarray((np.abs((output - input_img).squeeze().detach().cpu().numpy()).sum(axis=0))>0.1)
    mask = mask.resize((h,w), resample= Image.BILINEAR)
    mask_array = np.array(mask) 

    res = input_array.copy()
    mask_3d = np.repeat(mask_array[:, :, np.newaxis], 3, axis=2).astype(float)
    res = Image.fromarray((res * (1 - mask_3d) + upsample_output * mask_3d).astype(np.uint8))

    return res


def inference_multiprocessing(imgs, model, original_size, size_img=(192, 192), device='cpu', n_jobs=1):


    # downsampling_img_size = partial(downsampling, size_img=size_img)

    if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=n_jobs)

    # results = pool.map(downsampling_img_size, imgs)
    # size_imgs = [size_img]*n_jobs
    # print(size_imgs)
    results = pool.starmap(downsampling, zip(imgs, repeat(size_img)))

    pool.close()
    pool.join()


    input_img  = torch.cat([x[1] for x in results], dim=0)

    outputs = forward_model(input_img, model, device=device)

    upsampling_img = partial(uspampling, dim=original_size)

    inputs_arrays = [x[0] for x in results]
    input_imgs = [x[1] for x in results]
    outputs = [output for output in outputs]

    pool = multiprocessing.Pool(processes=n_jobs)


    results = pool.map(upsampling_img, zip(inputs_arrays, input_imgs, outputs))

    pool.close()
    pool.join()

    return results