'''
Title           :make_predictions_1.py
Description     :This script makes predictions using the 1st trained model and generates a submission file.
Author          :Adil Moujahid
Date Created    :20160623
Date Modified   :20160625
version         :0.2
usage           :python make_predictions_1.py
python_version  :2.7.11
'''

import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
import sys

#caffe python path
sys.path.append("D:\Lifatek_Technical\OYPC\caffe-windows\python")

import caffe
from caffe.proto import caffe_pb2

caffe.set_mode_gpu() 

#Size of images
IMAGE_WIDTH = 100
IMAGE_HEIGHT = 100

'''
Image processing helper function
'''

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


'''
Reading mean image, caffe model and its weights 
'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()
data = open("D:\\PeripheralBloodSmear\\CATDOGTutorial\\Image\\mean.binaryproto", "rb").read()
mean_blob.ParseFromString(data)

mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))


#Read model architecture and trained model's weights
net = caffe.Net('D:\\PeripheralBloodSmear\\CATDOGTutorial\\caffe_models\\googlenet_caffe\\train_val.prototxt',
                'D:\\PeripheralBloodSmear\\CATDOGTutorial\\caffe_models\\googlenet_caffe\\bvlc_googlenet_iter_40000.caffemodel',
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

'''
Making predicitions
'''
#Reading image paths
test_img_paths = [img_path for img_path in glob.glob("D:\\PeripheralBloodSmear\\DeepLearning\\Image\\test\\*jpg")]

#Making predictions
test_ids = []
preds = []
for img_path in test_img_paths:
	img = cv2.imread(img_path, cv2.IMREAD_COLOR)
	img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
	net.blobs['data'].data[...] = transformer.preprocess('data', img)
	out = net.forward()
	pred_probas = out['loss1/top-1']
	test_ids = test_ids + [img_path.split('/')[-1][:-4]]
	preds = preds + [pred_probas.argmax()]
	print img_path
	print preds
	print out

'''
Making submission file
'''
with open("D:\\PeripheralBloodSmear\\CATDOGTutorial\\Image\\submission_model_1.csv","w") as f:
    f.write("id,label\n")
    for i in range(len(test_ids)):
        f.write(str(test_ids[i])+","+str(preds[i])+"\n")
f.close()
