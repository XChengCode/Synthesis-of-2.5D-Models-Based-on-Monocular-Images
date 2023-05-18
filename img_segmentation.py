import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

def img_seg(layer_num = 3 , input_img = None, output_img = None):


    Pixel_RGB_List = output_img.reshape(output_img.shape[0]*output_img.shape[1],3)
    Pixel_RGB_List = np.float32(Pixel_RGB_List)

    print("Image segmentation in progress...")
    nclass = layer_num #The number of layers
    kmeans_clf = KMeans(n_clusters=nclass)
    kmeans_clf.fit(Pixel_RGB_List)
    ClfResult = kmeans_clf.labels_

    label_df = pd.DataFrame(ClfResult.tolist(),columns=["Label"])
    feature_df = pd.DataFrame(Pixel_RGB_List.tolist(),columns=["R","G","B"])
    Pixel_df = pd.concat([feature_df,label_df],axis=1)
    print("Image segmentation is done.\n")

    sorted_average=[]
    class_label = []
    class_average = kmeans_clf.cluster_centers_
    for i in range(len(class_average)):
        sorted_average.append(class_average[i][0])
        class_label.append(i)
    average_df = pd.concat([pd.DataFrame(sorted_average,columns=["Average"]),
                            pd.DataFrame(class_label,columns=["Old Labels"])],axis=1)
    sorted_average_df = average_df.sort_values(by='Average', ascending=True).reset_index().drop("index",axis=1)
    sorted_average_df = pd.concat([sorted_average_df,
                            pd.DataFrame(class_label,columns=["New Labels"])],axis=1)

    new_label_list = []
    count=1
    for old_label in Pixel_df["Label"]:
        if(count==len(Pixel_df)):
            print(f"Hierarchical sorting in progress: 100%",end="\r",flush=True)
        if(count%10000==0):
            print(f"Hierarchical sorting in progress: {round(count/len(Pixel_df)*100)}%",end="\r",flush=True)
            
        for index in range(len(sorted_average_df["Old Labels"])):
            if(old_label==sorted_average_df["Old Labels"][index]):
                new_label_list.append(sorted_average_df["New Labels"][index])
        count+=1
    Sorted_Pixel_df = pd.concat([Pixel_df,
                                pd.DataFrame(new_label_list,columns=["New Labels"])],axis=1)
    print("\nHierarchical sorting is done.")


    save_seg_img_list=[]
    save_mask_img_list=[]
    for m in range(nclass):
        #print(f"The {m+1} time:")
        Seg_img_list=[]
        mask_img_list=[]
        for i in range(len(Pixel_df)):
            if(i%1000==0):
                print(f"Mapping texture to layer {m+1}: {round(i/len(Pixel_df)*100)}%",end="\r",flush=True)
            if((i+1)==len(Pixel_df)):
                print(f"Mapping texture to layer {m+1}: 100%",end="\r",flush=True)
                
            #class No.0 is the layer at the end, the largest class number is the layer at the head.
            
            if(Sorted_Pixel_df["New Labels"][i]==m):
                Seg_img_list.append(input_img.reshape(input_img.shape[0]*input_img.shape[1],3)[i])
                mask_img_list.append(255)
            if(Sorted_Pixel_df["New Labels"][i]>m):
                Seg_img_list.append([255,255,255])
                mask_img_list.append(0)
            if(Sorted_Pixel_df["New Labels"][i]<m):
                Seg_img_list.append([0,0,0])
                mask_img_list.append(255)

        Seg_img_array = np.array(Seg_img_list).reshape(output_img.shape[0],output_img.shape[1],3)
        save_seg_img_list.append(Seg_img_array)
        
        Seg_mask_array = np.array(mask_img_list).reshape(output_img.shape[0],output_img.shape[1])
        save_mask_img_list.append(Seg_mask_array)
        
        print("\nDone")

    return save_seg_img_list, save_mask_img_list