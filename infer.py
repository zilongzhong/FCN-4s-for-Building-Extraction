import numpy as np
from PIL import Image
import collections
import caffe

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('/home/finoa/Caffe Models/22828930_15_1.tiff')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((87.70652426, 95.65976301, 94.12943263))
in_ = in_.transpose((2,0,1))

# load net
for i in range(16,18):
	caffe_model = "/home/finoa/Caffe Models/snapshot/FCN4s_building_1e-11_iter_%d.caffemodel" % ((i+1)*2000)
	net = caffe.Net('/home/finoa/Building-Extraction-FCN-4s/fcn.berkeleyvision.org/voc-fcn4S/deploy.prototxt', caffe_model, caffe.TEST)
	# shape for input (data blob is N x C x H x W), set data
	net.blobs['data'].reshape(1, *in_.shape)
	net.blobs['data'].data[...] = in_
	# run net and take argmax for prediction
	net.forward()
	out = (net.blobs['score'].data[0].argmax(axis=0)-1)*255
	#out2 = net.blobs['score_pool3c'].data[0].argmax(axis=0)

	print(out)
	print(out.shape)
	print(max(out.ravel()))
	print(collections.Counter(out.ravel()))

	#print(out2)
	#print(out2.shape)
	#print(max(out2.ravel()))
	#print(collections.Counter(out2.ravel()))

	img = Image.fromarray(out.astype(np.uint8))
	save_name = "/home/finoa/Caffe Models/fcn4s_22828930_15_1_%d.bmp" % ((i+1)*2000)
	img.save(save_name)

#/home/finoa/Caffe Models/snapshot/FCN4s_building_1e-11_iter_28000.caffemodel
