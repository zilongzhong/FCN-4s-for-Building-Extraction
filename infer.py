import numpy as np
from PIL import Image
import collections
import caffe

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('/home/finoa/Caffe Models/test01.tiff')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((87.70652426, 95.65976301, 94.12943263))
in_ = in_.transpose((2,0,1))

# load net

caffe_model = "/path/to/FCN4s_building.caffemodel" 
net = caffe.Net('/path/to/deploy.prototxt', caffe_model, caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = (net.blobs['score'].data[0].argmax(axis=0)-1)*255

print(out)
print(out.shape)
print(max(out.ravel()))
print(collections.Counter(out.ravel()))


img = Image.fromarray(out.astype(np.uint8))
save_name = "/path/to/model_predict.bmp" 
img.save(save_name)


