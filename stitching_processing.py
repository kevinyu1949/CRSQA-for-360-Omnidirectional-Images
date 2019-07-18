#!/usr/bin/env python
# coding: utf-8

# In[22]:


'''
function: Processing for Histogram, Perceptual Hash and 
            sparse coding reconstruction representation(stitching area)
result: the result will wirte into the txt that you can definit the specific path
'''

import cv2
import time
import random
import os,sys
import argparse
import numpy as np
from PIL import Image
from sklearn import metrics
from numpy import linalg as la
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.spatial.distance import cosine
from sklearn.decomposition import sparse_encode

def get_ROI_Patches(ROI_img):
    counter = 0
    key_patches = np.zeros(shape=(9842,8,8,3))
    height = 2127 
    width = 295 
    for row in range(0,height,8):
        for col in range(0,width,8):
            for i in range(0,8,1):
                for j in range(0,8,1):
                      key_patches[counter][i][j] = ROI_img[row+i][col+j]
            counter = counter+1
    return key_patches

def gaussian_random(center,besides):
    sigma_1 = random.sample(range(8,28),center) 
    sigma_2 = random.sample(range(0,7),besides)
    sigma_3 = random.sample(range(29,36),besides)
    random_list = []
    random_list.append(sigma_1+sigma_2+sigma_3)

    tempoary = str(random_list)
    tempoary = tempoary.replace('[','')
    tempoary = tempoary.replace(']','')
    random_list = list(eval(tempoary))
    random_list.sort()
    return random_list

# -------------Histogram statistics-------------------
def col_calcHis(img,savename):
    count_png = 0
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[16],[0,256])
    count_png += 1
    count_rang = ['0-15','16-31','32-47','48-63','64-79','80-95','96-111','112-127','128-143',
                  '144-159','160-175','176-191','192-207','208-223','224-239','240-255']
    for i in range(16):
        histr[i] = int(histr[i])
    return histr

# -------------Perceptual Hash algorithm--------------
def reduceimg(img):
    try:
        size = (64,64)
        img = Image.fromarray(img)
        img = img.resize(size, Image.ANTIALIAS)
        return img
    except IOError:
        raise Exception("Bad File")
        
def greyscale(img):
    img = img.convert('1')
    return img

def average_colors(img):
    pixelWeight = list(img.getdata())
    listLen = len(pixelWeight)
    totalsum = 0
    counter = 0
    for i in range(listLen):
        totalsum += pixelWeight[i]
        counter += 1
    averageVal = totalsum/counter
    return averageVal

def compare_bits(img,imgAvg):
    pixelWeight = list(img.getdata())
    listLen = len(pixelWeight)
    assert(listLen==4096)
    bitRes = ""
    for i in range(listLen):
        greyscale = rgb2grey(pixelWeight[i])
        if greyscale > imgAvg:
            bitRes += "1"
        else:
            bitRes += "0"
    return bitRes

def rgb2grey(rgbTuple):
    red = rgbTuple[0]
    green = rgbTuple[1]
    blue = rgbTuple[2]
    greyscale = 0.299*red + 0.587*green + 0.114*blue
    return greyscale

def hammingDifference(bitNum1, bitNum2):
    result = 0
    for index in range(len(bitNum1)):
        if (bitNum2[index]!=bitNum1[index]):
            result += 1
    return result

def perceptual_hash(img1,img2):
    img1 = img1
    img1 = reduceimg(img1)
    greyImg1 = greyscale(img1)
    imgAvg1 = average_colors(greyImg1)
    bitHash1 = compare_bits(img1,imgAvg1)

    img2 = img2
    img2 = reduceimg(img2)
    greyImg2 = greyscale(img2)
    imgAvg2 = average_colors(greyImg2)
    bitHash2 = compare_bits(img2,imgAvg2)

    dif = hammingDifference(bitHash1,bitHash2)
    rate_dif = 100-float(dif/4096)*100
    
    bitHash1 = list(map(int,bitHash1))
    bitHash2 = list(map(int,bitHash2))
    return bitHash1,bitHash2,rate_dif
# -------------Perceptual Hash algorithm-----------------

# =====================================[********Experiment********]============================================= #

def MainCode(datapath_0, datapath_180, datapath_90, datapath_270, fileName):
#     img_path_0 = '/home/kaiwenyu/ACMMM/算法实验对象-CVPR/Experi-images/8_街道/B/0796/' + fileName
#     img_path_180 = '/home/kaiwenyu/ACMMM/算法实验对象-CVPR/Experi-images/8_街道/B/0798/' + fileName
    img_path_0 = datapath_0 + fileName
    img_path_180 = datapath_180 + fileName
    img_0 = plt.imread(img_path_0)
    img_180 = plt.imread(img_path_180)
    scale_size = (5472,2736)
    num2img_0 = Image.fromarray(img_0)
    num2img_180 = Image.fromarray(img_180)
    num2img_0 = num2img_0.resize(scale_size, Image.ANTIALIAS)
    num2img_180 = num2img_180.resize(scale_size, Image.ANTIALIAS)
    
    img_0 = np.array(num2img_0)
    img_180 = np.array(num2img_180)
    img_region_bad_270 = img_0[304:2432, 1218:1518]
    img_region_bad_90 = img_0[304:2432, 3954:4254]

#     img_path_90 = "/home/kaiwenyu/ACMMM/算法实验对象-CVPR/Experi-images/8_街道/B/0799/" + fileName
#     img_path_270 = "/home/kaiwenyu/ACMMM/算法实验对象-CVPR/Experi-images/8_街道/B/0797/"+ fileName
    img_path_90 = datapath_90 + fileName
    img_path_270 = datapath_270 + fileName
    img_90 = plt.imread(img_path_90)
    img_270 = plt.imread(img_path_270)
    
    scale_size = (5472,2736)
    num2img_90 = Image.fromarray(img_90)
    num2img_270 = Image.fromarray(img_270)
    num2img_90 = num2img_90.resize(scale_size, Image.ANTIALIAS)
    num2img_270 = num2img_270.resize(scale_size, Image.ANTIALIAS)

    img_90 = np.array(num2img_90)
    img_270 = np.array(num2img_270)

    img_region_good_90 = img_90[304:2432, 2586:2886]
    img_region_good_270 = img_270[304:2432, 2586:2886]

    # Three channel histogram statistics features
    img_colhis_90_bad = col_calcHis(img_region_bad_90,'img_region_bad_90')
    img_colhis_270_bad = col_calcHis(img_region_bad_270,'img_region_bad_270')
    img_colhis_90_good = col_calcHis(img_region_good_90,'img_region_good_90')
    img_colhis_270_good = col_calcHis(img_region_good_270,'img_region_good_270')

    sum_90_good = 0
    sum_90_bad = 0
    difference = 0
    for i in range(len(img_colhis_90_bad)):
        sum_90_bad += img_colhis_90_bad[i]
        sum_90_good += img_colhis_90_good[i]
        difference += abs(img_colhis_90_good[i]-img_colhis_90_bad[i]) 

    sum_270_good = 0
    sum_270_bad = 0
    difference = 0
    for i in range(len(img_colhis_270_bad)):
        sum_270_bad += img_colhis_270_bad[i]
        sum_270_good += img_colhis_270_good[i]
        difference += abs(img_colhis_270_good[i]-img_colhis_270_bad[i]) 

    diff_90 = 100-float((difference/(sum_90_good+sum_270_good))*100)
    diff_270 = 100-float((difference/(sum_270_good+sum_90_good))*100)
    Hist_diff = (diff_90+diff_270)/2 

    # Perceptural Hash Algorithm
    bithash90_1,bithash90_2,difference_90 = perceptual_hash(img_region_good_90,img_region_bad_90)
    bithash270_1,bithash270_2,difference_270 = perceptual_hash(img_region_good_270,img_region_bad_270)
    Hash_diff = (difference_90+difference_270)/2 

    # Change to 227*32
    size = (32,227)
    height = 32; width = 227
    N_good = 20; N_bad = 14

    img_bad_90 = Image.fromarray(img_region_bad_90);img_bad_90 = img_bad_90.resize(size, Image.ANTIALIAS)
    img_bad_270 = Image.fromarray(img_region_bad_270);img_bad_270 = img_bad_270.resize(size, Image.ANTIALIAS)
    img_good_90 = Image.fromarray(img_region_good_90);img_good_90 = img_good_90.resize(size, Image.ANTIALIAS)
    img_good_270 = Image.fromarray(img_region_good_270);img_good_270 = img_good_270.resize(size, Image.ANTIALIAS)

    img_bad_90 = np.array(img_bad_90); img_bad_270 = np.array(img_bad_270)
    img_good_90 = np.array(img_good_90); img_good_270 = np.array(img_good_270)
    
    # Choose one of them
    # [1]
#     goodselect = [3,5,7,9,11,12,13,14,15,16,17,18,19,20,21,22,24,26,28,30]
#     badselect = [4,8,11,12,13,15,16,17,18,20,21,22,25,29]
    # [2]
    goodselect = []; badselect = []
    goodselect = random.sample(range(1, height), N_good)
    badselect = random.sample(range(1, height), N_bad)

    # Sampling patches by Gaussian alike distribution
    # Create fix size numpy arrays
    sample_bad_90 = np.zeros((width,N_bad,3));sample_bad_270 = np.zeros((width,N_bad,3))
    sample_good_90 = np.zeros((width,N_good,3));sample_good_270 = np.zeros((width,N_good,3))

    for i in range(len(sample_good_90)):
        count = 0
        for j in goodselect:
            sample_good_90[i][count] = img_good_90[i][j]
            sample_good_270[i][count] = img_good_270[i][j]
            count += 1

    for i in range(len(sample_bad_90)):
        count = 0
        for j in badselect:
            sample_bad_90[i][count] = img_bad_90[i][j]
            sample_bad_270[i][count] = img_bad_270[i][j]
            count += 1

    # Reshape and transpose
    sample_bad_90 = sample_bad_90.reshape((width, -1)); sample_bad_270 = sample_bad_270.reshape((width, -1))
    sample_good_90 = sample_good_90.reshape((width, -1)); sample_good_270 = sample_good_270.reshape((width, -1))

    sample_bad_90 = sample_bad_90.T; sample_bad_270 = sample_bad_270.T
    sample_good_90 = sample_good_90.T; sample_good_270 = sample_good_270.T

    # Just need sample_bad_90, sample_bad_270, sample_good_90, sample_good_270 
    # Stitching detection of the first image using two other cross-reference images. 
    start_time = time.clock()
    sparse_code_90 = sparse_encode(sample_bad_90, sample_good_90, n_jobs=1, check_input=False)
    sparse_code_270 = sparse_encode(sample_bad_270, sample_good_270, n_jobs=1,check_input=False)
    end_time = time.clock()

    U_90,Sigma_90,VT_90 = la.svd(sparse_code_90)
    U_270,Sigma_270,VT_270 = la.svd(sparse_code_270)

    sum_sigma_90 = np.sum(Sigma_90)
    sum_sigma_270 = np.sum(Sigma_270)

    sum_temp = 0
    for i in range(len(Sigma_90)):
        sum_temp = sum_temp + Sigma_90[i]
        if (sum_temp / sum_sigma_90) > 0.99 :
            break
    number_1 = i+1

    sum_temp = 0
    for i in range(len(Sigma_270)):
        sum_temp = sum_temp + Sigma_270[i]
        if (sum_temp / sum_sigma_270)>0.99 :
            break
    number_2 = i+1
    # Choose one of them
    Num_sparsecode = number_1 + number_2 # SparseCoding representation result
#   Num_sparsecode = (number_1 + number_2) / 2
    return Hist_diff, Hash_diff, Num_sparsecode
# ------------------------------------------------------------------------------ #

if __name__ == '__main__':
    time_start = time.time()
    
#   filepath = os.listdir() 
    filepath = ['1-samsung.jpg','dualfisheye.jpg','三脚架.jpg','体视.jpg','后期.jpg','等角.jpg','等距.jpg']
    txtpath = '****/sitich_features.txt' # Change path to your result txt
    f = open(txtpath, 'a')
    lens = len(fileName)
    
    # According to the specific sense to change the path
    datapath = ['/home/user/images/meeting room/A/0518', '/home/user/images/meeting room/A/0519',
            '/home/user/images/meeting room/A/0520', '/home/user/images/meeting room/A/0521']
    for i in range(lens):
        hist_diff, hash_diff, num_sparsecode = MainCode(datapath[0], datapath[1], datapath[2], 
                                                        datapath[3],filepath[i])
        f.writelines(str(hist_diff))
        f.writelines('  ')
        f.writelines(str(hash_diff))
        f.writelines('  ')
        f.writelines(str(num_sparsecode))
        f.writelines('\n')
    time_end = time.time()
    f.close()

