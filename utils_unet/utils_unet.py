import torch 
import numpy as np
from PIL import Image


def inference(img, model, path_save=None, size_img=(192, 192), device='cpu'):

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

    if path_save is not None:
        res.save(path_save)
    
    res = np.array(res)

    return res