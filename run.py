import numpy as np
import os
import shutil
import cv2
import PIL.Image
import torch
import torchvision.transforms as T
import argparse
import warnings
warnings.filterwarnings('ignore')

import depth_img_generator
from img_segmentation import img_seg
from Fixing_mask import generate_images
from RGBA_convertor import RGBA_Convertor


def empty_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' is not existing!")
        return
    
    
    try:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' is cleaned.")
    except Exception as e:
        print(f"Folder '{folder_path}' error: {e}")




def run(layer_num = 3, input_path = None, output_path = None, abs_path = ""):

    input_img = np.array(PIL.Image.open(input_path))

    output_img = depth_img_generator.run(input_img = input_img, 
                                         model_path = abs_path + "weights/dpt_beit_large_512.pt", 
                                         model_type='dpt_beit_large_512')

    original_shape = input_img.shape
    resize_input_img = cv2.resize(input_img,dsize=(512,512), interpolation=cv2.INTER_CUBIC)
    resize_output_img = cv2.resize(output_img,dsize=(512,512), interpolation=cv2.INTER_CUBIC)

    save_seg_img_list, save_mask_img_list = img_seg(layer_num=layer_num, input_img=resize_input_img, output_img=resize_output_img)

    process_img_list=[]
    for img in save_seg_img_list:
        process_img_list.append(img.transpose(2,0,1))

    Fixed_img_list = generate_images(network_pkl=abs_path+"weights/Places_512_FullData.pkl",
                                     image_list=process_img_list,
                                     mask_list=save_mask_img_list)
    Transparent_img_list = RGBA_Convertor(Fixed_img_list)

    final_list=[]
    transform = T.ToPILImage()
    for img in Transparent_img_list:
        tensor_img = torch.tensor(img).permute(2,0,1)
        resize_img = transform(tensor_img/255).resize((original_shape[1],original_shape[0]))
        final_list.append(np.array(resize_img))

    print(f"Saving images into folder...")

    empty_folder(output_path)

    for i in range(len(final_list)):
        image = PIL.Image.fromarray(final_list[i])
        image.save(output_path + '/picture_'+ str(i) +'.png')
        
    print(f"Finish")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--layer_num',
                        type=int, default=3,
                        help='Number of layers.'
                        )


    parser.add_argument('-i', '--input_path',
                        default=None,
                        help='Folder with input images (if no input path is specified, images are tried to be grabbed '
                             'from camera)'
                        )

    parser.add_argument('-o', '--output_path',
                        default=None,
                        help='Folder for output images'
                        )

    parser.add_argument('--abs_path',
                        default="",
                        help='Absolute path'
                        )

    args = parser.parse_args()
    run(args.layer_num, args.input_path, args.output_path, args.abs_path)
