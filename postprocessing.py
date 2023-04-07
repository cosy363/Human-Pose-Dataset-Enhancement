from module import *
from postprocessing import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def post_processing(img):


    #GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    rc = (0,0,img.shape[0],img.shape[1])
    cv2.grabCut(img, mask, rc, None, None, 5, cv2.GC_INIT_WITH_RECT)
    # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    mask2 = np.where((mask==0),0,1).astype('uint8')
    new_img = img*mask2[:,:,np.newaxis]
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)

    #

    return new_img