from module import *
from postprocessing import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image

###Input
test_mode = 0 # 0: Test All / 1: Test only selected mode / 2: 
select_mode = 0
view_toggle = True
clean_output_folder = True

image_folder = 'Input/Image/'
mask_folder = 'Input/Mask/'
seg_folder = 'Input/Segmentation/'
output_folders = ['Output/1.segANDmask/','Output/2.IFsegANDmask/','Output/3.segORmask/','Output/4.segANDmaskANDpost/','Output/5.seg/','Output/6.mask/']

#Clean Output folder
if clean_output_folder is True:
    list = output_folders
    for output_folder in list:
        for f in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, f))    


#Part 0. Define Import of Image and Mask from folder
class Dataset:
    def __init__(
            self, 
            image_path, 
            anno_path, 
            seg_path,
    ):
        self.ids = os.listdir(image_path)
        self.ids2 = os.listdir(anno_path)
        self.ids4 = os.listdir(seg_path)
        self.images_fps = [os.path.join(image_path, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(anno_path, image_id2) for image_id2 in self.ids2]
        self.segs_fps = [os.path.join(seg_path, image_id4) for image_id4 in self.ids4]

    
    def __getitem__(self, i):
        
        # read data
        images_fps = self.images_fps
        masks_fps = self.masks_fps
        segs_fps = self.segs_fps
        ids = self.ids2

        #remove ds_store
        for j,data in enumerate(images_fps):
            if data.find('.DS_Store') != -1:
                del images_fps[j]
                
        for j,data in enumerate(masks_fps):
            if data.find('.DS_Store') != -1:
                del masks_fps[j]

        for j,data in enumerate(segs_fps):
            if data.find('.DS_Store') != -1:
                del segs_fps[j]        

        for j,data in enumerate(ids):
            if data.find('.DS_Store') != -1:
                del ids[j]                    

        image = cv2.imread(images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(masks_fps[i], 0)
        seg = cv2.imread(segs_fps[i], 0)

        id = ids[i]


        return image, mask, seg, id

    def __len__(self):
        return len(self.ids)



#Part 1. Mode Implementation
dataset = Dataset(image_folder, mask_folder, seg_folder)


# visualize(
#     image = dataset[0][0],
#     mask = dataset[0][1],
#     gt = dataset[0][2],
#     seg = dataset[0][3],
# )

img, mask, seg, ids = [],[], [], []
for i in dataset:
    cvt = cv2.cvtColor(i[2], cv2.COLOR_BGR2RGB)
    seg_img = Image.fromarray(cvt)
    seg_img = seg_img.resize((i[0].shape[1],i[0].shape[0]),Image.LANCZOS)
    seg_img = cv2.cvtColor(np.array(seg_img),cv2.COLOR_BGR2GRAY)
    seg.append(cv2.bitwise_not(seg_img))

    cvt2 = cv2.cvtColor(i[1], cv2.COLOR_BGR2RGB)
    msk_img = Image.fromarray(cvt2)
    msk_img = msk_img.resize((i[0].shape[1],i[0].shape[0]),Image.LANCZOS)
    msk_img = cv2.cvtColor(np.array(msk_img),cv2.COLOR_BGR2GRAY)
    mask.append(msk_img)

    # mask.append(i[1])
    ids.append(i[3])
    img.append(i[0])

# Mode 1: Seg ∩ Mask
print("MODE 1")
mode1 = []
for n,data in enumerate(seg):
    mode1_img = cv2.bitwise_and(seg[n], mask[n], mask=None)
    for i in range(mode1_img.shape[0]):
        for j in range(mode1_img.shape[1]):            
            if mode1_img[i][j] > 0:
                mode1_img[i][j] == 255
    mode1.append(mode1_img)

list_save_images(output_folders[0], mode1, ids)


# Mode 2: If iou < 95%, then Seg ∩ Mask. else Mask
print("MODE 2")
mode2 = []
for n,data in enumerate(seg):
    if iou_score(seg[n],mask[n]) < 0.90:
        mode2_img = cv2.bitwise_and(seg[n], mask[n], mask=None)
        for i in range(mode2_img.shape[0]):
            for j in range(mode2_img.shape[1]):            
                if mode2_img[i][j] > 0:
                    mode2_img[i][j] == 255
        mode2.append(mode2_img)
    else:
        mode2.append(mask[n])

list_save_images(output_folders[1], mode2, ids)

# Mode 3: Seg ∪ Mask
print("MODE 3")

mode3 = []
for n,data in enumerate(seg):
    mode3_img = cv2.bitwise_or(seg[n], mask[n], mask=None)
    for i in range(mode3_img.shape[0]):
        for j in range(mode3_img.shape[1]):            
            if mode3_img[i][j] > 0:
                mode3_img[i][j] == 255
    mode3.append(mode3_img)

list_save_images(output_folders[2], mode3, ids)

# Mode 4: Seg ∩ Mask ∩ Post-processing
print("MODE 4")
mode4 = []
for n,data in enumerate(seg):
    segmask = cv2.bitwise_and(seg[n], mask[n], mask=None)

    masked_img = img[n]
    init_GT = (seg[n]>0)    # 값이 있으면 1, 없으면 0
    outputs=[[],[],[]]
    for i in range(3):
        output = masked_img[:,:,i] * init_GT
        outputs[i] = output 
    outputs = np.transpose(outputs,(1,2,0))

    # mode4_img = cv2.bitwise_and(segmask, mask[n], mask=None)
    try:
        mode4_img = cv2.bitwise_and(masked_img, post_processing(outputs), mask=None)
        mode4_img = (mode4_img>0).long
        mode4_img = cv2.bitwise_and(segmask, post_processing(img[n]), mask=None)
        for i in range(mode4_img.shape[0]):
            for j in range(mode4_img.shape[1]):            
                if mode4_img[i][j] > 0:
                    mode4_img[i][j] == 255
        mode4.append(mode4_img)
    except:
        mode4.append(segmask)
    print(n)
    

list_save_images(output_folders[3], mode4, ids)


# Mode 5: Seg Only
print("MODE 5")

mode5 = []
for img in seg:
    mode5_img = img
    for i in range(mode5_img.shape[0]):
        for j in range(mode5_img.shape[1]):            
            if mode5_img[i][j] > 0:
                mode5_img[i][j] == 255
    mode5.append(mode5_img)
list_save_images(output_folders[4], mode5, ids)

# Mode 6: Mask Only
print("MODE 6")

mode6 = []
for data in dataset:
    mode6_img = data[1]
    for i in range(mode6_img.shape[0]):
        for j in range(mode6_img.shape[1]):            
            if mode6_img[i][j] > 0:
                mode6_img[i][j] == 255    
    mode6.append(mode6_img)
list_save_images(output_folders[5], mode6, ids)


