import torch
import numpy as np
import random
import dnnlib
import legacy
import os

from networks.mat import Generator

def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def generate_images(
    network_pkl="weights/Places_512_FullData.pkl",
    image_list=None,
    mask_list=None,
    resolution=512,
    truncation_psi=1,
    noise_mode='const',
    outdir="test_output",
):
    fixing_result_list = []
    seed = 240  # pick up a random number
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if mask_list is not None:
        assert len(image_list) == len(mask_list), 'illegal mapping'

    print(f'Loading networks from: {network_pkl}')
    device = torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G_saved = legacy.load_network_pkl(f)['G_ema'].to(device).eval().requires_grad_(False) # type: ignore
    net_res = 512 if resolution > 512 else resolution
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(device).eval().requires_grad_(False)
    copy_params_and_buffers(G_saved, G, require_all=True)

    os.makedirs(outdir, exist_ok=True)

    # no Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    if resolution != 512:
        noise_mode = 'random'
    with torch.no_grad():
        for i in range(len(image_list)):
            print(f"Fixing the missing area of image {i+1}",end="\n",flush=True)
            image = (torch.from_numpy(image_list[i]).float().to(device) / 127.5 - 1).unsqueeze(0)

            if mask_list is not None:

                mask = mask_list[i].astype(np.float32) / 255.0
                mask = torch.from_numpy(mask).float().to(device).unsqueeze(0).unsqueeze(0)

            z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
            output = G(image, mask, z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
            output = output[0].cpu().numpy()
            #output = cv2.resize(output,dsize=(image_shape[1],image_shape[0]), interpolation=cv2.INTER_CUBIC)
            #PIL.Image.fromarray(output, 'RGB').save(f'{outdir}/{i}')
            fixing_result_list.append(output)
        print(f"{i+1} images have been fixed.")
            
    return fixing_result_list
