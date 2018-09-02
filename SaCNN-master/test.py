#!/usr/bin/env python
# -*- coding: utf-8 -*-

caffe_root = '/home/zll/OPT/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import os
import caffe
import numpy as np

root='/home/zll/OPT/SaCNN-CrowdCounting-Tencent_Youtu/SaCNN-master/'
model_def=root+'/val.prototxt'
model_weights=root+'/model/mine/allpretrain_iter_120000.caffemodel'#change the caffemodel name
net = caffe.Net(model_def,
                model_weights,
                caffe.TEST)

log_path = root+'/model/mine/testalllog.txt'
imgs = os.listdir(root+'ShanghaiTech/Part_B/test_dataall/images')
imgNum = len(imgs)
MSE=0.0
MAE=0.0
file = open(log_path,mode='a')
file.write('total test images are '+str(imgNum)+'\n'+'MAE      '+'MSE'+'\n')
print 'processing......'
for i in range(imgNum):
	img = root + 'ShanghaiTech/Part_B/test_data/images/IMG_'+str(i+1)+'.jpg'
	out=net.forward()
	gt_count = net.blobs['GTCount'].data
	estd_count = net.blobs['EstdCount'].data
	MAE += np.abs(gt_count-estd_count)
	MSE += ((gt_count-estd_count)*(gt_count-estd_count))

MAE = MAE/imgNum
MSE = np.sqrt(MSE/imgNum)

file.write(str(MAE)+' '+str(MSE)+'\n')
file.close()
print('Done!')