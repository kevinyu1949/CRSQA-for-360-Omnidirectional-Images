#!/usr/bin/env python
# coding: utf-8

# In[8]:


'''
function: Processing for color chromatism and blind zone(global area)
result: the result will wirte into the txt that you can definit the specific path
'''

import cv2 # OpenCV version == 3.1.0
import numpy as np
from matplotlib import pyplot as plt

# Color chromatism
def col_diff(path_stitchImg,path_fisheyeImg):
    img1 = cv2.imread(path_stitchImg) # queryImage
    img2 = cv2.imread(path_fisheyeImg) # trainImage

    img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1_gray,None)
    kp2, des2 = sift.detectAndCompute(img2_gray,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    matchesMask = [[0,0] for i in range(len(matches))] # draw only good matches

    count = 0 # Count the number of points matched
    num = 1000 # The number is up to you, may you can choose 2000, 4000 or others
    keypoints1 = np.zeros((num,2)) # according to condition change size
    keypoints2 = np.zeros((num,2))
    
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            # coordinate row and col ara adverse
            keypoints1[count][1] = int(kp1[i].pt[0])
            keypoints1[count][0] = int(kp1[i].pt[1])
            keypoints2[count][1] = int(kp2[i].pt[0])
            keypoints2[count][0] = int(kp2[i].pt[1])
            count += 1
        if count == num:
            break

    draw_params = dict(matchColor = (0,255,0),singlePointColor = (255,0,0),matchesMask = matchesMask,flags = 0)

    img_1 = path_stitchImg
    img_2 = path_fisheyeImg
    image_1 = plt.imread(img_1);image_2 = plt.imread(img_2)

    sum_pixel = 0 # sum of matching point pixel value
    for x,y in keypoints1:
        x = int(x);y = int(y)
        if x != 0 and y != 0:
            sum_channel = 0
            for i in range(3):
                sum_channel = sum_channel + image_1[x][y][i]
            sum_pixel = sum_pixel + sum_channel
    average_pixel = sum_pixel / (3*count)

    sum_pixel_2 = 0
    for x,y in keypoints2:
        x = int(x);y = int(y)
        if x != 0 and y != 0:
            sum_channel_2 = 0
            for i in range(3):
                sum_channel_2 = sum_channel_2 + image_2[x][y][i]
            sum_pixel_2 = sum_pixel_2 + sum_channel_2
    average_pixel_2 = sum_pixel_2 / (3*count)

    score_coldiff = round(100 * (1 - abs(average_pixel - average_pixel_2) / 256),4)
    return score_coldiff

# Blind zone
def bli_zone(filename):
    '''
    In the future, we will add the scheme that determine the blind zone score by the 
    proportion of the size of the blind zone
    '''
    filepath = ['/1-samsung.jpg','/dualfisheye.jpg','/三脚架.jpg','/体视.jpg','/后期.jpg','/等角.jpg','/等距.jpg']
    value_by_person = [7, 5, 6, 4, 2, 2, 2]
    lens = len(filepath)
    dic_blizone = []
    
    for i in range(lens):
        dic_blizone[filepath[i]] =  value_by_person[i]
        
    for key, value in dicts.items():
        if key == filename:
            return value

if __name__ == '__main__':
    base_path = '/home/user/images/meeting room/A/0518/'
    pathfisheye = '360_0518.JPG'; 
    base = pathfisheye.split('.')[0].split('_')[1]
    filepath = ['/1-samsung.jpg','/dualfisheye.jpg','/三脚架.jpg','/体视.jpg','/后期.jpg','/等角.jpg','/等距.jpg']
    
    filetxt = '****/global_features.txt' # Change to your result txt
    print('-----rate of color difference-----')
    for img in filepath:
        pathstitchingimg = base + img
        coldiff = col_diff(base_path + pathstitchingimg, base_path + pathfisheye)
        blizone = bli_zone(img)
        with open(filetxt, 'a') as f:
            f.write(coldiff)
            f.write(blizone)

