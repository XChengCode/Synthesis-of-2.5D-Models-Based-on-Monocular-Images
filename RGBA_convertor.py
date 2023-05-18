import numpy as np
import cv2

def RGBA_Convertor(img_list=None):

    print(f"Converting into RGBA images...")

    RGBA_img_list=[]
    for img in img_list:
        alpha_channel = np.ones(img.shape[:2], dtype=img.dtype) * 255
        img_rgba = cv2.merge((img, alpha_channel))
        RGBA_img_list.append(img_rgba)

    Transparent_img_list=[]
    for img in RGBA_img_list:
        img_with_hole_list=[]
        for height in img:
            for width in height:
                if(width[0]==0)and(width[1]==0)and(width[2]==0):
                    img_with_hole_list.append([0,0,0,0])
                else:
                    img_with_hole_list.append(width)
        img_with_hole_array = np.array(img_with_hole_list).reshape(img.shape[0],img.shape[1],4)
        Transparent_img_list.append(img_with_hole_array)
        
    return Transparent_img_list