#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
function: Using multiple linear regression to fit the rule of stitching image quality assessment
result: the result will show the relative score(or rank) for each stitched image.

for example:
X = [[48, 61.44, 85.00, 95.86, 7],
     [51, 41.17, 78.85, 94.51, 5],
     [52, 57.80, 83.40, 92.59, 6],
     [41, 38.28, 79.14, 98.66, 4],
     [53, 58.71, 82.18, 92.24, 2],
     [32, 44.96, 71.36, 98.46, 2],
     [37, 41.77, 77.77, 97.76, 2]]

Y = [6,4,5,2,0,1,3]
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize

'''
The feature txt contain the stitch and global features,
stitch txt contain three numbers, global txt contain two numbers,
join these together as the input data
'''
stitch_feature_txt = '****/stitch_feature.txt'
global_feature_txt = '****/global_feature.txt'

with open(stitch_feature_txt, 'r') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip().split(' ')
        print(lines[i])
    for i in range(len(lines)): # str -> int
        for j in range(len(lines[0])):
            lines[i][j] = int(lines[i][j])
    stitch_feature = lines

with open(global_feature_txt, 'r') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip().split(' ')
        print(lines[i])
    for i in range(len(lines)): # str -> int
        for j in range(len(lines[0])):
            lines[i][j] = int(lines[i][j])
    global_feature = lines
    
input_data = np.hstack(stitch_feature, global_feature)
input_height = input_data.shape[0]

'''
According to the pair-wise compare result to fill, and you can 
divide the data set into a train and test part.
'''
Y = np.array([[***],[***]]) 
for i in range(input_height):
    Y[i] = normalize(Y[i], axis=0, norm='max')
    
# Normalize the input data
print('X.shape:',X.shape)
print('Y.shape:',Y.shape)

tmp_x = X; tmp_y = Y
trans_matrix_x = np.reshape(tmp_x, (-1,7,5))
print('trans_matrix:', trans_matrix_x.shape)

counter_x = trans_matrix_x[0]
counter_y = trans_matrix_y[0]

for i in range(counter):
    trans_matrix[i] = normalize(trans_matrix[i], axis=0, norm='max')
# print(trans_matrix)
print('trans_matrix_normalize:',trans_matrix.shape)
x_normalize = np.reshape(trans_matrix, (counter_y,5))
print('x_normalize:',x_normoalize.shape)
print('x_normalize:\n',x_normalize)

print('-'*40)
trans_matrix_y = np.reshape(tmp_y, (-1,7))

for i in range(counter_x):
    trans_matrix_y = normalize(trans_matrix_y, axis=1, norm='max')
    
y_normalize = trans_matrix_y
y_normalize = np.reshape(y_normalize, (counter_y,))
print('y_normalize:\n',y_normalize)
print(x_normalize.shape,y_normalize.shape)

x_train = x_normalize
y_train = y_normalize

# train linear regression
model = LinearRegression()
model.fit(x_train,y_train)

# test or predict linear regression. Following is an example.
x_predict = np.array([[67, 69.69, 91.93, 99.66, 7],
                      [62, 53.80, 87.38, 97.78, 5],
                      [66, 65.93, 88.02, 94.24, 6],
                      [61, 54.12, 84.59, 96.58, 4],
                      [66, 62.51, 87.27, 94.19, 2],
                      [62, 55.56, 81.67, 93.93, 2],
                      [65, 56.07, 85.86, 94.21, 2]])
x_predict = normalize(x_predict, axis=0, norm='max')
y_pred = model.predict(x_predict)
print('predict result:{}:'.format(y_pred))

