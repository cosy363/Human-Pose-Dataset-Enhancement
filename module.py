from module import *
from postprocessing import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(15, 6))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def denormalize(x):
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

def iou_score(im1, im2):
    overlap = (im1>0.5) * (im2>0.5)
    union = (im1>0.5) + (im2>0.5)
    return overlap.sum()/float(union.sum())

def mean_iou_score(list1,list2):
    count,score = 0,0
    for i,j in zip(list1,list2):
        score += iou_score(i,j)
        count += 1
    return (score/count)

def iou_list(list1,list2):
    list = []
    for i,j in zip(list1,list2):
        list.append(iou_score(i,j))
    return list
        
def dataset_save_images(output_folder, dataset,target):
    for data in dataset:
        directory = output_folder + data[4]
        cv2.imwrite(directory,data[target])

def list_save_images(output_folder, list, ids):
    for n,id in enumerate(ids):
        target = output_folder + id
        cv2.imwrite(target,list[n])
