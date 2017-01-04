# Fully convolutional networks for building and road extraction

This is the reference implementation of the codes and models for the FCN-4s in the IGARSS2016 FCN paper:<br />
'''
[Fully convolutional networks for building and road extraction: Preliminary results](FCN for building and road extraction.pdf)<br />
Zilong Zhong*, Jonathan Li, Weihong Cui, Han Jiang<br />
IGARSS 2016<br />
'''
Before deploying this model, make sure you have installed the latest [BVLC/caffe:master.](https://github.com/BVLC/caffe)<br />

Futher exploration about FCN models please refer to [FCN for Segmentation.](https://github.com/shelhamer/fcn.berkeleyvision.org)<br />
'''
Fully Convolutional Models for Semantic Segmentation<br />
Jonathan Long*, Evan Shelhamer*, Trevor Darrell<br />
CVPR 2015<br />
arXiv:1411.4038<br />
'''
Training command:<br />
    python TrainingVal/solve.py<br />
Evaluating command:<br />
    python evaluate.py<br />