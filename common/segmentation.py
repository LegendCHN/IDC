#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import scipy.misc
import time
import argparse
import sys

caffe_root = 'CAFFE ROOT'


sys.path.insert(0, caffe_root + 'python')
import caffe

#this window size needs to be exactly the same size as that used to extract the patches from the matlab version
wsize = 32
hwsize= int(wsize/2)

MODEL_FILE = '/workspace/model/deploy_train32.prototxt'
PRETRAINED = '/workspace/model_resize/1_caffenet_train_w32_iter_600000.caffemodel'
IMAGE= '/workspace/test2.png'  

#load our mean file and reshape it accordingly
a = caffe.io.caffe_pb2.BlobProto()
file = open('DB_train_resize_w32_1.binaryproto' ,'rb')
data = file.read()
a.ParseFromString(data)
means = a.data
means = np.asarray(means)
means = means.reshape(3, 32, 32)

#make sure we use teh GPU otherwise things will take a very long time
caffe.set_mode_gpu()

#load the model
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=means,
                       channel_swap=(2, 1, 0),
                       raw_scale=255,
                       image_dims=(32, 32))

newfname_class = "test_class.png"
newfname_prob = "test_prob.png" 

image = caffe.io.load_image(IMAGE)
#mirror the edges so that we can compute the full image
image = np.lib.pad(image, ((hwsize, hwsize), (hwsize, hwsize), (0, 0)), 'symmetric') 
outputimage_probs = np.zeros(shape=(image.shape[0],image.shape[1],3)) #make the output files where we'll store the data
outputimage_class = np.zeros(shape=(image.shape[0],image.shape[1]))
linenumber = 0
for rowi in xrange(hwsize+1,image.shape[0]-hwsize):
	patches=[] #create a set of patches, oeprate on a per column basis
	print('processing line number %s',linenumber)
	for coli in xrange(hwsize+1,image.shape[1]-hwsize):
		patches.append(image[rowi-hwsize:rowi+hwsize, coli-hwsize:coli+hwsize,:])
	linenumber += 1
	prediction = net.predict(patches) #predict the output 
	pclass = prediction.argmax(axis=1) #get the argmax
	outputimage_probs[rowi,hwsize+1:image.shape[1]-hwsize,0:2]=prediction #save the results to our output images
	outputimage_class[rowi,hwsize+1:image.shape[1]-hwsize]=pclass
		
outputimage_probs = outputimage_probs[hwsize:-hwsize, hwsize:-hwsize, :] #remove the edge padding
outputimage_class = outputimage_class[hwsize:-hwsize, hwsize:-hwsize]

scipy.misc.imsave(newfname_prob,outputimage_probs) #save the files
scipy.misc.imsave(newfname_class,outputimage_class)
