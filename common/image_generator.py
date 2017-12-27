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

#parse the command line arguments
parser = argparse.ArgumentParser(description='Generate full image outputs from Caffe DL models.')
parser.add_argument('base',help="Base Directory type {nuclei, epi, mitosis, etc}")
parser.add_argument('fold',type=int,help="Which fold to generate outputfor")
args= parser.parse_args()

#this window size needs to be exactly the same size as that used to extract the patches from the matlab version
wsize = 32
#hwsize= int(wsize/2)

BASE=args.base
FOLD=args.fold

#Locations of the necessary files are all assumed to be in subdirectoires of the base file
MODEL_FILE = '/model/1-alexnet_traing_32w_db.prototxt' 
PRETRAINED = '/model/%d_caffenet_train_w32_iter_700000.caffemodel' % (FOLD)
IMAGE_DIR= '/subs/' 
OUTPUT_DIR= '/logs/' 

#if our output directory doesn't exist, lets create it. each fold gets a numbered directory inside of the image directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#load our mean file and reshape it accordingly
a = caffe.io.caffe_pb2.BlobProto()
file = open('DB_train_w32_%d.binaryproto' % (FOLD) ,'rb')
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

#see which files we need to produce output for in this fold
#we look at the parent IDs in the test file and only compute those images
#as they've been "held out" from the training set
files=open('test_w32_parent_%d.txt'%(FOLD),'rb')
positives = 0
negatives = 0
true_positives = 0
true_negatives = 0
false_postives = 0
false_negatives = 0

#go into the image directory so we can use glob a bit easier
os.chdir(IMAGE_DIR)

for base_fname in files:
	base_fname=base_fname.strip()
	for fname in sorted(glob.glob("%s_*.png"%(base_fname))): #get all of the files which start with this patient ID
		print fname
		if fname[-6] == '0':
			negatives += 1
		elif fname[-6] == '1':
			postives += 1

		image = caffe.io.load_image(fname) #load our image
		prediction = net.predict(image) #predict the output 
		pclass = prediction.argmax(axis=1) #get the argmax

		if (fname[-6] == '0') and (pclass == int(fname[-6])):
			true_negatives += 1
		elif (fname[-6] == '0') and (pclass != int(fname[-6])):
			false_negatives += 1
		elif (fname[-6] == '1') and (pclass == int(fname[-6])):
			true_positives += 1
		elif (fname[-6] == '1') and (pclass != int(fname[-6])):
			false_postives += 1
#28314
#74330
#20040
#68723
#8274
#5607
#precision = TP/(TP+FP) 0.708
#recall = TP/ (TP+FN) 0.781
#0.743
print(postives)
print(negatives)
print(true_positives)
print(true_negatives)
print(false_postives)
print(false_negatives)