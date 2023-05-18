import torch
import cv2
import numpy as np

from midas.model_loader import load_model

def process(device, model, model_type, image, input_size, target_size, optimize):
    
    sample = torch.from_numpy(image).to(device).unsqueeze(0)
    if optimize and device == torch.device("cuda"):
        sample = sample.to(memory_format=torch.channels_last)
        sample = sample.half()
    height, width = sample.shape[2:]
    prediction = model.forward(sample)
    prediction = (torch.nn.functional.interpolate(prediction.unsqueeze(1),
                                                  size=target_size[::-1],
                                                  mode="bicubic",
                                                  align_corners=False).squeeze().cpu().numpy())
    return prediction


def run(input_img, model_path, model_type="dpt_beit_large_512", optimize=False, side=False, height=None,
        square=False, grayscale=False):

    print("Initialize")

    # select device
    device = torch.device("cpu")
    print("Device: %s" % device)

    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize)

    # get input
    print("Start processing")

    # input in [0, 1]
    image = transform({"image": input_img / 255.0})["image"]

    # compute
    with torch.no_grad():
        prediction = process(device, model, model_type, image, (net_w, net_h), input_img.shape[1::-1],optimize)
        
    depth_min = prediction.min()
    depth_max = prediction.max()
    max_val = (2**(8*1))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (prediction - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(prediction.shape, dtype=prediction.dtype)

    out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)
    out = cv2.cvtColor(out.astype("uint8"), cv2.COLOR_BGR2RGB)

    print("Finished")
    return out
